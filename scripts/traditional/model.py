# Traditional machine learning approach using a Hidden Markov Model

import pandas as pd
import re
import random
import numpy as np
import sys
import os
import json
import pickle

# Handle potential import errors with scikit-learn
ML_AVAILABLE = True
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Warning: Machine learning imports failed: {e}")
    print("Running in fallback mode with rule-based classification only")
    ML_AVAILABLE = False


# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importing Data
from scripts.etl.extract import load_data


class HMMAdvisor:
    """
    Advisor system combining HMM for state tracking and ML for classification.
    Tracks emotional states and problem types to provide relevant responses.
    """

    def __init__(self, load_pretrained=True):
        """Sets up the advisor with dataset or pre-trained models."""
        # Only load data if we need to train models
        if load_pretrained and os.path.exists('models/emotion_classifier.pkl'):
            self.data = None  # Don't load data if not needed
        else:
            # Loading data for training
            self.data = load_data()
            print("Available columns:", self.data.columns.tolist())
        
        # Our states and categories
        self.emotions = ['anxiety', 'depression', 'sadness', 'anger', 'fear', 'happiness']
        self.problems = ['job crisis', 'ongoing depression', 'breakup with partner',
                         'problems with friends', 'academic pressure']

        # Response types we'll use
        self.strategies = ['Question', 'Reflection', 'Suggestion', 'Information', 'Reassurance']

        # Train or load our models
        self._train_classifiers()

        # Set up the HMM
        self._build_transition_matrix()
        self._build_emission_matrix()

        # Setting up our canned responses
        self._create_response_templates()

        # Init tracking vars
        self.current_emotion = None
        self.current_problem = None
        self.last_strategy = None

    def _train_classifiers(self):
        """Trains our emotion and problem classifiers from the dataset, or loads from serialized files."""
        # Setting flag for ML availability
        self.ml_models_ready = False
        
        # If ML libraries aren't available, skip training
        if not ML_AVAILABLE:
            print("ML libraries not available - using rule-based classification only")
            return
            
        try:
            # Check if serialized models exist
            if os.path.exists('models/emotion_classifier.pkl') and os.path.exists('models/problem_classifier.pkl'):
                try:
                    # Load serialized models instead of retraining
                    serializer = joblib if 'joblib' in sys.modules else pickle
                    self.emotion_vectorizer = serializer.load('models/emotion_vectorizer.pkl')
                    self.emotion_classifier = serializer.load('models/emotion_classifier.pkl')
                    self.problem_vectorizer = serializer.load('models/problem_vectorizer.pkl')
                    self.problem_classifier = serializer.load('models/problem_classifier.pkl')
                    print("Loaded pre-trained models from disk")
                    self.ml_models_ready = True
                    return
                except Exception as e:
                    print(f"Error loading models: {e}")
                    print("Falling back to training new models")
                
            # If models don't exist or couldn't be loaded, load data and train
            parsed_df = self._parse_json_data()
            
            # Train emotion classifier
            self.emotion_vectorizer, self.emotion_classifier = self._train_single_classifier(
                parsed_df, 'situation', 'emotion_type', self.emotions, MultinomialNB()
            )
            
            # Train problem classifier
            self.problem_vectorizer, self.problem_classifier = self._train_single_classifier(
                parsed_df, 'situation', 'problem_type', self.problems, RandomForestClassifier(n_estimators=100)
            )
            
            # Serialize the models for future use
            try:
                os.makedirs('models', exist_ok=True)
                serializer = joblib if 'joblib' in sys.modules else pickle
                serializer.dump(self.emotion_vectorizer, 'models/emotion_vectorizer.pkl')
                serializer.dump(self.emotion_classifier, 'models/emotion_classifier.pkl')
                serializer.dump(self.problem_vectorizer, 'models/problem_vectorizer.pkl')
                serializer.dump(self.problem_classifier, 'models/problem_classifier.pkl')
                print("Models trained and serialized to disk")
                self.ml_models_ready = True
            except Exception as e:
                print(f"Error saving models: {e}")
        except Exception as e:
            print(f"Error in training classifiers: {e}")
            print("Using rule-based classification only")

    def _parse_json_data(self):
        """Parse the JSON in 'text' column to actual data."""
        parsed_data = []
        for _, row in self.data.iterrows():
            try:
                if 'text' in row:
                    json_data = json.loads(row['text'])
                    parsed_data.append(json_data)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error parsing row: {e}")
                continue
        
        # Create a new DataFrame with the parsed data
        parsed_df = pd.DataFrame(parsed_data)
        print("Parsed columns:", parsed_df.columns.tolist())
        
        return parsed_df

    def _train_single_classifier(self, df, feature_col, target_col, valid_labels, classifier_instance):
        """Helper method to train a single classifier."""
        # Extract features and target
        X = df[feature_col].astype(str).fillna('') if feature_col in df.columns else pd.Series([''] * len(df))
        y = df[target_col].astype(str).fillna('unknown') if target_col in df.columns else pd.Series(['unknown'] * len(df))
        
        # Filter valid labels
        mask = y.isin(valid_labels)
        X_valid = X[mask]
        y_valid = y[mask]
        
        print(f"Training {target_col} classifier with {len(X_valid)} samples")
        
        # Vectorize features
        vectorizer = TfidfVectorizer(max_features=3000)
        X_tfidf = vectorizer.fit_transform(X_valid)
        
        # Train classifier
        classifier_instance.fit(X_tfidf, y_valid)
        
        return vectorizer, classifier_instance

    def _build_transition_matrix(self):
        """Creates transition probs between emotional states."""
        n_states = len(self.emotions)
        self.transition_matrix = np.zeros((n_states, n_states))

        # People tend to stay in emotional states, so higher self-transition
        for i in range(n_states):
            for j in range(n_states):
                if i == j:  # Same state
                    self.transition_matrix[i, j] = 0.7
                else:  # Different state
                    self.transition_matrix[i, j] = 0.3 / (n_states - 1)

        # Equal probs for starting state
        self.initial_probs = np.ones(n_states) / n_states

    def _build_emission_matrix(self):
        """
        Creates strategy distribution matrices for each emotion-problem combo.
        This helps pick good strategies based on emotional state.
        """
        # Based on some CBT literature and my own experience
        self.strategy_matrix = {
            'anxiety': {
                'Question': 0.2,
                'Reflection': 0.2, 
                'Suggestion': 0.3,  # Anxious people often want solutions
                'Information': 0.1,
                'Reassurance': 0.2
            },
            'depression': {
                'Question': 0.1,
                'Reflection': 0.2,
                'Suggestion': 0.2,
                'Information': 0.1,
                'Reassurance': 0.4  # Reassurance helps depression more
            },
            'sadness': {
                'Question': 0.1,
                'Reflection': 0.3,
                'Suggestion': 0.1,
                'Information': 0.1,
                'Reassurance': 0.4
            },
            'anger': {
                'Question': 0.2,
                'Reflection': 0.3,  # Reflection helps with anger
                'Suggestion': 0.2,
                'Information': 0.1,
                'Reassurance': 0.2
            },
            'fear': {
                'Question': 0.2,
                'Reflection': 0.2,
                'Suggestion': 0.3,
                'Information': 0.1,
                'Reassurance': 0.2
            },
            'happiness': {
                'Question': 0.3,
                'Reflection': 0.3,
                'Suggestion': 0.1,
                'Information': 0.1,
                'Reassurance': 0.2
            }
        }

        # Tweaks based on problem type
        self.problem_strategy = {
            'job crisis': {
                'Suggestion': 0.4,  # Practical stuff for job issues
                'Information': 0.2
            },
            'ongoing depression': {
                'Reassurance': 0.4,  # Support for depression
                'Reflection': 0.3
            },
            'breakup with partner': {
                'Reflection': 0.3,
                'Reassurance': 0.3
            },
            'problems with friends': {
                'Question': 0.3,
                'Suggestion': 0.3
            },
            'academic pressure': {
                'Suggestion': 0.4,  # Practical advice for school
                'Information': 0.2
            }
        }

    def _create_response_templates(self):
        """Sets up response templates - I've collected these from various sources."""
        # Main templates by strategy type
        self.templates = {
            'Question': [
                "How are you feeling about that?",
                "What's been hardest for you?",
                "How has this affected your day-to-day?",
                "When did you first notice this happening?",
                "What have you tried already?",
                "What would be a tiny step forward for you?",
                "Who's been there for you during this?",
                "What do you think would help most right now?"
            ],
            'Reflection': [
                "Sounds like you're feeling [EMOTION] about this whole thing.",
                "I can tell this has been rough on you.",
                "You're dealing with some big challenges right now.",
                "That must be really [EMOTION] to go through.",
                "I hear that this [PROBLEM] has hit you hard.",
                "Makes sense you'd feel that way with what you're facing.",
                "This situation has clearly stirred up a lot for you.",
                "You've had a lot on your plate lately."
            ],
            'Suggestion': [
                "Maybe try breaking this down into smaller bits?",
                "Could be worth talking to someone who specializes in this.",
                "Sometimes a daily routine helps when things get crazy.",
                "Don't forget to take care of yourself during stressful times.",
                "Have you tried writing down your thoughts?",
                "Setting tiny goals sometimes helps get momentum going.",
                "Talking to others who've been there might give you some ideas.",
                "Maybe focus on the parts you can actually control for now."
            ],
            'Information': [
                "Lots of people struggle with [PROBLEM] - you're definitely not alone.",
                "It's really common for [PROBLEM] to bring up [EMOTION] feelings.",
                "From what I've seen, mixing different coping strategies works best.",
                "These situations usually do improve with time and support.",
                "Those feelings are totally normal for what you're going through.",
                "Most people need both emotional and practical help during these times.",
                "This kind of stuff can mess with both your emotions and physical health.",
                "Just so you know, progress isn't usually straight - ups and downs happen."
            ],
            'Reassurance': [
                "What you're feeling makes total sense given what's happening.",
                "You're definitely not the only one who's dealt with this.",
                "Takes guts to talk about this stuff.",
                "You've shown a lot of strength dealing with this situation.",
                "This hard period won't last forever, even if it feels endless now.",
                "Taking it one day at a time is completely reasonable.",
                "It's fine to not have everything figured out right now.",
                "Even baby steps are still steps forward."
            ]
        }

        # Problem-specific responses that I've found helpful
        self.problem_templates = {
            'job crisis': {
                'Suggestion': [
                    "Might help to update your resume and ping your network",
                    "Setting a job search schedule could make it less overwhelming",
                    "Maybe use this time to think about what you really want next",
                    "Could be a good time to pick up a new skill to add to your resume",
                    "Breaking job hunting into daily mini-tasks helps a ton"
                ],
                'Reassurance': [
                    "Job stuff is tough, but often leads somewhere unexpected and good",
                    "Your job isn't who you are",
                    "Even super successful people get fired or laid off sometimes",
                    "It's ok to feel crappy about this before jumping into action",
                    "This sucks, but your skills and experience haven't disappeared"
                ]
            },
            'ongoing depression': {
                'Suggestion': [
                    "Talking to a professional might give you more tools to work with",
                    "Even tiny self-care habits can help chip away at this",
                    "Moving your body a bit, even just a walk, sometimes helps",
                    "What's the smallest possible win you could go for today?",
                    "Tracking your mood might help you spot patterns"
                ],
                'Reassurance': [
                    "Depression is a health issue, not a personality flaw",
                    "Recovery zigzags - good days and bad days are part of it",
                    "Your feelings are real, and there's no timeline for feeling better",
                    "People find their way through this, even when it feels impossible",
                    "Just getting through the day takes massive effort with depression"
                ]
            },
            'breakup with partner': {
                'Suggestion': [
                    "Creating new routines helps after a breakup",
                    "Might help to reconnect with friends or hobbies you've missed",
                    "It's OK to tell people you need a break from talking about it",
                    "Taking time for yourself can actually help the healing",
                    "Some people find physical activity helps process the emotions"
                ],
                'Reassurance': [
                    "The pain does fade with time, I promise",
                    "Everything you're feeling is a normal part of breakups",
                    "This relationship ending doesn't define you or your future",
                    "Healing happens in fits and starts - it's messy",
                    "Feeling this bad shows you're capable of real connection"
                ]
            },
            'problems with friends': {
                'Suggestion': [
                    "Sometimes a calm, direct conversation clears things up",
                    "Writing out your thoughts first can help you say what you mean",
                    "Setting clearer boundaries might prevent similar issues",
                    "Taking a little space from the friendship can give perspective",
                    "Lean on your other solid relationships right now"
                ],
                'Reassurance': [
                    "Even good friendships hit rough patches",
                    "Your feelings about this are totally valid",
                    "Setting boundaries is healthy, even if it feels awkward",
                    "This fight doesn't erase all the good parts of your friendship",
                    "You deserve friends who treat you with respect"
                ]
            },
            'academic pressure': {
                'Suggestion': [
                    "Breaking assignments into micro-tasks makes them less scary",
                    "Try scheduling both work blocks and breaks to stay balanced",
                    "Talking to professors about what's going on might help",
                    "Study groups can make the work more bearable",
                    "Try the 25-minute focus technique - it works for a lot of people"
                ],
                'Reassurance': [
                    "You're worth more than your grades",
                    "Plenty of successful people bombed classes or even whole semesters",
                    "This intense period will end, even though it feels never-ending",
                    "Taking care of yourself is just as important as your academics",
                    "Struggling with certain subjects doesn't mean you're not smart"
                ]
            }
        }

        # These come from my experience with different emotional states
        self.emotion_templates = {
            'anxiety': [
                "I can hear the wheels spinning in your head from here.",
                "Sounds like your brain is in overdrive with worry.",
                "That feeling when anxiety takes the wheel is brutal.",
                "Living with that constant worry is exhausting, huh?",
                "Anxiety likes to make us imagine the worst possible scenarios."
            ],
            'depression': [
                "That heavy, stuck feeling of depression is the worst.",
                "Sounds like you've got that numbness that comes with depression.",
                "Depression has this way of making everything look hopeless.",
                "Even getting out of bed can feel like climbing a mountain when depression hits.",
                "Depression just drains every ounce of energy and motivation."
            ],
            'sadness': [
                "I can hear how sad you're feeling about this.",
                "This loss has clearly hit you hard.",
                "That sadness makes sense - this matters to you.",
                "The depth of your sadness shows how meaningful this was.",
                "You're going through a real grieving process here."
            ],
            'anger': [
                "I can feel the frustration coming through your words.",
                "This situation has clearly pissed you off, and with good reason.",
                "Your anger makes sense - boundaries were crossed.",
                "That burning feeling when someone treats you unfairly...",
                "The anger seems totally justified given what happened."
            ],
            'fear': [
                "I can hear the fear in how you're describing this.",
                "This has clearly triggered some deep fear for you.",
                "That stomach-dropping feeling of fear is awful.",
                "Fear kicks in when we're facing unknowns or threats.",
                "The fear seems to be coloring how you see everything right now."
            ],
            'happiness': [
                "I can feel your excitement coming through!",
                "This has clearly brought you some real joy.",
                "Your happiness about this is contagious.",
                "Love the positive energy as you talk about this.",
                "You sound genuinely pumped about this!"
            ]
        }

    def detect_emotion(self, message):
        """Figure out the emotion using ML and keyword backups."""
        # Try ML if available and models are ready
        if ML_AVAILABLE and hasattr(self, 'ml_models_ready') and self.ml_models_ready:
            try:
                message_vec = self.emotion_vectorizer.transform([message])
                emotion = self.emotion_classifier.predict(message_vec)[0]

                # If we got something valid, use it
                if emotion in self.emotions:
                    return emotion
            except Exception as e:
                print(f"ML emotion detection error: {e}")
                # Continue to keyword fallback
        
        # Keyword matching (as fallback or primary if ML not available)
        message = message.lower()

        # Define emotion keywords for better matching
        emotion_keywords = {
            'happiness': ["happy", "good", "great", "excellent", "joy", "excited", "delighted", "thrilled", "pleased"],
            'sadness': ["sad", "unhappy", "crying", "tears", "grief", "blue", "down", "heartbroken", "upset"],
            'anger': ["angry", "mad", "frustrated", "annoyed", "furious", "irritated", "pissed", "outraged", "hate"],
            'anxiety': ["anxious", "worried", "stress", "nervous", "uneasy", "concerned", "panicked", "overthinking"],
            'fear': ["scared", "afraid", "terrified", "frightened", "fearful", "horror", "dread", "panic"],
            'depression': ["depressed", "hopeless", "empty", "worthless", "numb", "meaningless", "pointless"]
        }

        # Count keyword matches for each emotion
        emotion_scores = {emotion: 0 for emotion in self.emotions}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    emotion_scores[emotion] += 1
        
        # Get the emotion with the highest score
        max_score = max(emotion_scores.values())
        if max_score > 0:
            # If there's a tie, prioritize more common emotions
            max_emotions = [e for e, s in emotion_scores.items() if s == max_score]
            if len(max_emotions) == 1:
                return max_emotions[0]
            else:
                # Prioritize anxiety and depression as they're more common
                for priority in ["anxiety", "depression", "sadness"]:
                    if priority in max_emotions:
                        return priority
                return max_emotions[0]
        
        # When all else fails, most people are anxious
        return "anxiety"

    def detect_problem(self, message):
        """Figure out the problem type using ML and keywords."""
        # Try ML if available and models are ready
        if ML_AVAILABLE and hasattr(self, 'ml_models_ready') and self.ml_models_ready:
            try:
                message_vec = self.problem_vectorizer.transform([message])
                problem = self.problem_classifier.predict(message_vec)[0]

                # If valid, run with it
                if problem in self.problems:
                    return problem
            except Exception as e:
                print(f"ML problem detection error: {e}")
                # Continue to keyword fallback
        
        # Define more comprehensive keyword sets for each problem type
        message = message.lower()
        
        problem_keywords = {
            'job crisis': [
                "job", "work", "career", "fired", "employer", "boss", "office", 
                "workplace", "unemployed", "interview", "resume", "layoff", "promotion"
            ],
            'ongoing depression': [
                "depressed", "depression", "therapy", "medication", "mental health", 
                "psychiatrist", "psychologist", "hopeless", "empty", "worthless",
                "counseling", "treatment"
            ],
            'breakup with partner': [
                "breakup", "ex", "relationship", "partner", "boyfriend", "girlfriend", 
                "husband", "wife", "divorce", "dating", "romance", "split", "dumped"
            ],
            'problems with friends': [
                "friend", "friendship", "social", "betrayed", "trust", "bestie", 
                "buddy", "group", "clique", "hanging out", "ghosted", "toxic"
            ],
            'academic pressure': [
                "school", "college", "exam", "study", "class", "university", "grade", 
                "professor", "teacher", "assignment", "homework", "degree", "graduate"
            ]
        }

        # Count keyword matches for each problem type
        problem_scores = {problem: 0 for problem in self.problems}
        
        for problem, keywords in problem_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    problem_scores[problem] += 1
        
        # Get the problem with the highest score
        max_score = max(problem_scores.values())
        if max_score > 0:
            max_problems = [p for p, s in problem_scores.items() if s == max_score]
            if len(max_problems) == 1:
                return max_problems[0]
            else:
                # In case of a tie, prioritize based on common prevalence
                for priority in ["ongoing depression", "job crisis", "breakup with partner"]:
                    if priority in max_problems:
                        return priority
                return max_problems[0]
        
        # Most common issue in our data if no keywords match
        return "ongoing depression"

    def update_emotional_state(self, detected_emotion):
        """Update the HMM state with new emotion info."""
        if not self.current_emotion:
            # First message, just set it directly
            self.current_emotion = detected_emotion
            return

        # Get our indices
        current_idx = self.emotions.index(self.current_emotion)
        detected_idx = self.emotions.index(detected_emotion)

        # Check transition probability
        trans_prob = self.transition_matrix[current_idx, detected_idx]

        # Roll the dice on whether to change state
        if random.random() < trans_prob:
            self.current_emotion = detected_emotion

    def select_strategy(self):
        """Pick which approach to use based on our emission matrix."""
        if not self.current_emotion:
            return "Question"  # Default if we're flying blind

        # Get base distribution for this emotion
        strategy_dist = self.strategy_matrix[self.current_emotion]

        # Tweak based on problem type if we know it
        if self.current_problem and self.current_problem in self.problem_strategy:
            problem_dist = self.problem_strategy[self.current_problem]

            # Blend the distributions
            blended_dist = strategy_dist.copy()
            for strategy, prob in problem_dist.items():
                blended_dist[strategy] = (blended_dist.get(strategy, 0) + prob) / 2

            # Make sure it still adds to 1
            total = sum(blended_dist.values())
            strategy_dist = {k: v/total for k, v in blended_dist.items()}

        # Try not to repeat ourselves
        if self.last_strategy:
            # Cut down the odds of the same strategy again
            temp_dist = strategy_dist.copy()
            if self.last_strategy in temp_dist:
                temp_dist[self.last_strategy] *= 0.5

            # Re-normalize if needed
            total = sum(temp_dist.values())
            if total > 0:
                temp_dist = {k: v/total for k, v in temp_dist.items()}
                strategy_dist = temp_dist

        # Pick based on our weighted distribution
        strategies = list(strategy_dist.keys())
        probs = [strategy_dist[s] for s in strategies]

        return random.choices(strategies, weights=probs, k=1)[0]

    def generate_response(self, strategy, message):
        """Build a response based on our templates and context."""
        # Pick the right template type
        if strategy == "Reflection" and self.current_emotion and self.current_emotion in self.emotion_templates:
            # Use emotion-specific reflections
            template = random.choice(self.emotion_templates[self.current_emotion])
        elif (strategy in ["Suggestion", "Reassurance"] and
              self.current_problem and
              self.current_problem in self.problem_templates and
              strategy in self.problem_templates[self.current_problem]):
            # Use problem-specific templates for these
            template = random.choice(self.problem_templates[self.current_problem][strategy])
        else:
            # Fall back to general templates
            template = random.choice(self.templates[strategy])

        # Fill in our placeholders
        template = template.replace("[EMOTION]", self.current_emotion or "concerned")
        template = template.replace("[PROBLEM]", self.current_problem or "situation")

        return template

    def respond(self, message):
        """Process what the user said and respond appropriately."""
        # Catch empty messages
        if not message or message.strip() == "":
            return "I'm here whenever you want to talk.", {}

        # Figure out what's going on
        detected_emotion = self.detect_emotion(message)
        detected_problem = self.detect_problem(message)

        # Update our model
        self.update_emotional_state(detected_emotion)
        self.current_problem = detected_problem

        # Pick our approach
        strategy = self.select_strategy()
        self.last_strategy = strategy

        # Create a response
        response = self.generate_response(strategy, message)

        # Include some debug stuff
        debug_info = {
            'emotion': self.current_emotion,
            'problem': self.current_problem,
            'strategy': strategy
        }

        return response, debug_info


# Interactive testing
def run_advisor():
    """Starting an interactive session for testing"""
    advisor = HMMAdvisor()

    print("\n=== HMM Advisor ===")
    print("AI: Hey there. What's on your mind today?")
    print("(Type 'exit', 'quit', or 'bye' to end)")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("AI: Take care. Catch you later")
            break

        response, debug_info = advisor.respond(user_input)
        print(f"AI: {response}")
        print(f"Emotion: {debug_info['emotion']}")
        print(f"Problem: {debug_info['problem']}")
        print(f"Strategy: {debug_info['strategy']}")

# Run this if executed directly
if __name__ == "__main__":
    run_advisor()