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

load_dotenv()

# Initialize models
hmm_advisor = HMMAdvisor()

llm = Llama.from_pretrained(
    repo_id="haran-nallasivan/meowth-nlp-demo-0.1_llama-3.2-3b-instruct_q5_k_m_gguf",
    filename="model.gguf",
    verbose=False,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0,
    offload_kqv=True
)

# Define model generation functions
def generate_naive(messages):
    return naive_generation(messages)

def generate_traditional(messages):
    return hmm_advisor.respond(messages[-1]['content'])[0]

def generate_deep(messages):
    messages_with_system_prompt = [{
        'role': 'system',
        'content': SYSTEM_PROMPT
    }]
    messages_with_system_prompt.extend(messages)
    response = llm.create_chat_completion(
        messages=messages_with_system_prompt,
        temperature=0.7,
        max_tokens=100,
        stop=["</s>"],
    )
    return response['choices'][0]['message']['content']

# Model selection
available_models = [generate_naive, generate_deep, generate_traditional]

def evaluation_by_criteria_ref_free(prompt):
    criteria_list = ("Technical Accuracy, Structural Adherence, Empathetic Tone, Intervention depth, Clinical safety").split(',')
    results = []

    for model in available_models:
        print(f"Evaluating model: {model.__name__}")
        response = model([{
            'role': 'user',
            'content': prompt
        }])
        print(f"Model Response: {response}")

        eval_prompt = criteria_based_evaluation_prompt.format(patient_prompt=prompt, response=response)
        eval_result = generate_naive([{"role": "user", "content": eval_prompt}])
        results.append(eval_result)

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

def main():
    prompt = input("Enter your prompt: ")
    evaluation_results = evaluation_by_criteria_ref_free(prompt)
    print("Evaluation Results:")
    print(json.dumps(evaluation_results, indent=2))

if __name__ == "__main__":
    main()