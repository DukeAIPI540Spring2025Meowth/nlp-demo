from openai import OpenAI
from llama_cpp import Llama
from prompts import criteria_based_evaluation_prompt
import json
from dotenv import load_dotenv
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.naive.model import generation as naive_generation
from scripts.traditional.model import HMMAdvisor
from scripts.deep.transform import SYSTEM_PROMPT
from scripts.eval.extract_test_split import extract_test_split
from scripts.deep.transform import transform_to_openai

load_dotenv()

class Evaluator:

    def __init__(self):
        # Initialize models
        self.hmm_advisor = HMMAdvisor()

        self.llm_base = Llama.from_pretrained(
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="Llama-3.2-3B-Instruct-Q5_K_M.gguf",
            verbose=False,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            offload_kqv=True
        )
        self.llm = Llama.from_pretrained(
            repo_id="haran-nallasivan/meowth-nlp-demo-0.1_llama-3.2-3b-instruct_q5_k_m_gguf",
            filename="model.gguf",
            verbose=False,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            offload_kqv=True
        )
        
    # Define model generation functions
    def generate_naive(self, messages):
        return naive_generation(messages)

    def generate_traditional(self, messages):
        return self.hmm_advisor.respond(messages[-1]['content'])[0]
    
    def generate_base_llma(self, messages):
        messages_with_system_prompt = [{
            'role': 'system',
            'content': SYSTEM_PROMPT
        }]
        messages_with_system_prompt.extend(messages)
        response = self.llm_base.create_chat_completion(
            messages=messages_with_system_prompt,
            temperature=0.7,
            max_tokens=100,
            stop=["</s>"],
        )
        return response['choices'][0]['message']['content']
    
    def generate_deep(self, messages):
        messages_with_system_prompt = [{
            'role': 'system',
            'content': SYSTEM_PROMPT
        }]
        messages_with_system_prompt.extend(messages)
        response = self.llm.create_chat_completion(
            messages=messages_with_system_prompt,
            temperature=0.7,
            max_tokens=100,
            stop=["</s>"],
        )
        return response['choices'][0]['message']['content']

    def evaluation_by_criteria_ref_free(self, prompt, ground_truth_response):
        # Model selection
        available_models = [self.generate_naive, self.generate_base_llma, self.generate_deep, self.generate_traditional]
        results = {}

        for model in available_models:
            print(f"Evaluating model: {model.__name__}")
            response = model([{
                'role': 'user',
                'content': prompt
            }])
            print(f"Model Response: {response}")

            eval_prompt = criteria_based_evaluation_prompt.format(patient_prompt=prompt, ground_truth_response=ground_truth_response, response=response)
            eval_result = self.generate_naive([{"role": "user", "content": eval_prompt}])
            results[model.__name__] = json.loads(eval_result)

            # results[model.__name__] = {}
            # for cri in criteria_list:
            #     eval_prompt = criteria_based_evaluation_prompt.format(patient_prompt=prompt, criteria=cri, response=response)
            #     eval_result = model([{"role": "user", "content": eval_prompt}])
            #     if eval_result:
            #         results[model.__name__][cri] = eval_result
            #         print(f"Criteria: {cri}")
            #         print(f"Score: {eval_result.get('score', 'N/A')}/5")
            #         print(f"Explanation: {eval_result.get('explanation', 'N/A')}")
            #         print("\n")
            #     else:
            #         print(f"Failed to evaluate {cri} for model {model.__name__}.\n")

        return results

def average_evaluation_scores(evaluation_results):
    """
    Averages the scores for generate_naive, generate_deep, and generate_traditional across all criteria.

    Args:
        evaluation_results (list): A list of dictionaries containing evaluation results for each model.

    Returns:
        dict: A dictionary with average scores for each model per criterion.
    """
    # Initialize a dictionary to hold cumulative scores and counts for each model and criterion
    cumulative_scores = {
        'generate_naive': {},
        'generate_base_llama': {},
        'generate_deep': {},
        'generate_traditional': {}
    }

    # Iterate through each evaluation result
    for result in evaluation_results:
        for model_name, criteria_scores in result.items():
            if model_name in cumulative_scores:
                for criterion, score in criteria_scores.items():
                    if criterion not in cumulative_scores[model_name]:
                        cumulative_scores[model_name][criterion] = {'total_score': 0, 'count': 0}
                    cumulative_scores[model_name][criterion]['total_score'] += score
                    cumulative_scores[model_name][criterion]['count'] += 1

    # Calculate average scores
    average_scores = {}
    for model_name, criteria_data in cumulative_scores.items():
        average_scores[model_name] = {}
        for criterion, data in criteria_data.items():
            if data['count'] > 0:
                average_scores[model_name][criterion] = data['total_score'] / data['count']
            else:
                average_scores[model_name][criterion] = 0  # Handle case with no scores

    return average_scores

def main():
    # Load the dataset and transform it
    extract_test_split()
    input_file = "esconv/test.json"  # Assuming this is the path to the original dataset
    output_file = "esconv/test_openai.json"
    transform_to_openai(input_file, output_file)

    # Load the transformed data
    with open(output_file, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)

    evaluator = Evaluator()
    results = []

    iter = 0
    max_iter = 200

    # Iterate over each conversation in the transformed data
    for item in evaluation_data:
        conversations = item['conversations']
        for i in range(1, len(conversations) - 1, 2):
            user_prompt = conversations[i]['content']
            assistant_response = conversations[i + 1]['content']
            
            # Get evaluation results for each pair
            eval_result = evaluator.evaluation_by_criteria_ref_free(user_prompt, assistant_response)
            results.append(eval_result)

            iter += 1

            if iter == max_iter:
                break
        if iter == max_iter:
            break


    print(results)
    print('\n')
    print(average_evaluation_scores(results))
    # Average the results (assuming eval_result is a list of scores)
    #average_results = {key: sum(result[key] for result in results) / len(results) for key in results[0].keys()}

    #print("Average Evaluation Results:")
    #print(json.dumps(average_results, indent=2))

if __name__ == "__main__":
    main()