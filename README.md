# MindHeal Assistant

A multi-approach emotional support conversation system using the [Emotional Support Conversations (esconv) dataset](https://github.com/thu-coai/Emotional-Support-Conversation).[[1]](#1)

## Overview

This project implements three approaches to emotional support conversations:
1. **Naive Approach**: Using a foundation model without special prompting, RAG, or finetuning
2. **Traditional ML Approach**: Hidden Markov Model (HMM)
3. **Deep Learning Approach**: Finetuned Llama-3.2-3B-Instruct model

## Live Demo

The application is deployed on Digital Ocean: [https://mindheal-assistant-7kfky.ondigitalocean.app/](https://mindheal-assistant-7kfky.ondigitalocean.app/)

## Running the Streamlit App

### Local Setup

```bash
# Navigate to the streamlit directory
cd streamlit

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Using Docker

```bash
# Build the Docker image locally
docker build -t mindheal-assistant:latest .

# Run the container
docker run -p 8501:8501 mindheal-assistant:latest
```

The app will be available at http://localhost:8501

### Pushing to Docker Hub

```bash
# Tag the image with your Docker Hub username. Leaving the tagname as 'tagname' was a mistake, but it's too late.
docker tag mindheal-assistant:latest yourusername/mindheal-assistant:tagname

# Login to Docker Hub
docker login

# Push the image to Docker Hub
docker push haranku16/mindheal-assistant:tagname
```

### Features
- Speech-to-text capability (requires API key)
- Response revision through ChatGPT (requires API key)
- Interactive conversation interface

## Setup for other scripts

```bash
# Navigate to the project root if you're not already there
cd ..

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running the Naive Approach

```bash
cd scripts/naive
python naive_approach.py
```

## Running the Finetuned Model

Navigate to the `deep` directory:

```bash
cd scripts/deep

# Set up environment variables
cp .env_example .env
# Edit .env with your API keys and configurations

# Run the finetuning pipeline
./pipeline.sh

# Run the example script
python example.py
```

## Running the HMM Advisor

Navigate to the `traditional` directory:

```bash
cd scripts/traditional

# Run the HMM Advisor
python hmm_advisor.py
```

## Running Evaluation

Navigate to the `eval` directory:

```bash
cd scripts/eval

# Set up environment variables
cp .env_example .env
# Edit .env with your configurations

# Run evaluation
python evaluate.py
```

## Technical Details

### Fine-tuning
- Used torchtune with Low Rank Adaptation (LoRA) recipe for Llama-3.2-3B-Instruct
- Used LORA configuration (3B_lora_single_device.yaml) copied and adapted with `tune copy` command
- Training performed on Google Colab with A100 GPU
- Fine-tuned for 5 epochs (~4-5 minutes per epoch)
- Converted to GGUF format using llama-cpp
- Applied quantization for model optimization to be runnable on CPUs

### Hidden Markov Model (HMM)
- Combines HMM for emotion state tracking with ML classifiers for emotion and problem detection
- Uses TF-IDF vectorization with MultinomialNB for emotion classification
- Employs RandomForest classifier for problem type categorization
- Implements transition matrices between emotional states based on therapeutic progression
- Maintains a library of response templates for different strategies (Question, Reflection, Suggestion, Information, Reassurance)
- Response selection determined by current emotional state and conversation context

### Evaluation (LLM-as-a-judge)
- Implements a criteria-based evaluation framework using an LLM as a judge
- Evaluates responses based on five key metrics:
  1. Technical Accuracy (1-5): Application of proper therapeutic techniques
  2. Structural Adherence (1-5): Following the ABCDE model in responses
  3. Empathetic Tone (1-5): Level of emotional validation vs. robotic phrasing
  4. Intervention Depth (1-5): Quality of follow-up questioning
  5. Clinical Safety (1-5): Detection of risk factors and implementation of proper protocols
- Compares performance across all three approaches (naive, traditional, and deep learning)

### Dataset
We used the [esconv dataset](https://huggingface.co/datasets/giliit/esconv), a crowd-sourced collection of emotional support conversations between therapists and patients.

## Ethical Considerations

This application is designed for educational purposes and should not replace professional mental health services. The responses generated are based on machine learning models and may not always provide appropriate or helpful guidance. Users should seek professional help for serious mental health concerns.

Because the finetuning and HMM approaches utilize the esconv dataset, they will reflect inherent biases within the dataset. Because the conversations are anonymized, we do not know the demographics of the therapists and patients whose conversations have been recorded. As a result, we cannot know if the dataset contains a diverse population of both therapists and patients across protected characteristics under federal law.

The system does not store conversations, and any API keys provided by users are only used for the specified services during the active session.

## Presentation

For more information, check our [presentation](https://docs.google.com/presentation/d/14s7PdyqKt8x5M0Cs7gqhqigjk0nqaL-D7-k_5IybErw/edit?usp=sharing).

## Citation

<a id="1">[1]</a> Liu, et al. (2021). [Toward Emotional Support Dialog Systems.](https://arxiv.org/abs/2106.01144) ACL.