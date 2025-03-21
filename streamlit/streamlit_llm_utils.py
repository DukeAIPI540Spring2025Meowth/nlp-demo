# This utility script manages interactions with two different language models:
# 1) A locally-hosted fine-tuned Llama model used as the main response generator,
#    providing initial conversational responses based on user input.
# 2) An external OpenAI GPT model ("judge") used to refine the original responses
#    from the Llama model and assign them a numeric quality score.
#
# The script defines clear initialization functions for both models:
# - The Llama model is loaded from a specified Hugging Face repository and configured locally.
# - The OpenAI judge model requires an API key and is initialized conditionally.
#
# Additionally, it provides functionality to generate responses with the main LLM,
# as well as refining and evaluating responses with the judge LLM.
# The judge produces a structured output containing both the refined response and
# a numeric score reflecting the effectiveness of the original response.

import os
from llama_cpp import Llama
from openai import OpenAI

# LOCAL LLAMA FOR MAIN MODEL  
def init_main_llm():
    """
    Initialize the main LLM (local Llama model).
    """
    return Llama.from_pretrained(
        repo_id="haran-nallasivan/meowth-nlp-demo-0.1_llama-3.2-3b-instruct_q5_k_m_gguf",
        filename="model.gguf",
        verbose=False,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0,
        offload_kqv=True
    )

def generate_main_response(llm, conversation):
    """
    Generate a response from the local Llama-based LLM, 
    given a list of chat messages in the 'conversation'.
    """
    response = llm.create_chat_completion(
        messages=conversation,
        temperature=0.7,
        max_tokens=200,
        stop=["</s>"]
    )
    return response["choices"][0]["message"]["content"]

# OPENAI FOR THE JUDGE MODEL   
def init_judge_llm(api_key: str):
    """
    Initialize the OpenAI client if the API key is provided.
    """
    if not api_key:
        print("Warning: OpenAI API key not yet provided.")
        return None  # No client created if key not provided
    
    client = OpenAI(api_key=api_key)
    return client

def judge_refine_and_score(judge_llm, user_input, candidate_response):
    judge_prompt = [
        {
            "role": "system",
            "content": (
                "You are an editor and evaluator. You will refine the candidate response, "
                "then rate how effectively the original response addresses the user's input "
                "on a scale from 1 to 10. Provide your answer strictly as follows:\n\n"
                "Refined: <your refined text>\n"
                "Score: <numeric score (1–10)>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User input: {user_input}\n\n"
                f"Candidate response: {candidate_response}\n\n"
                "Refine and score this candidate response."
            ),
        },
    ]

    judge_reply = judge_llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=judge_prompt,
        temperature=0.3,
        max_tokens=300
    )

    full_text = judge_reply.choices[0].message.content.strip()

    refined_text, score = "", "N/A"
    for line in full_text.split("\n"):
        line = line.strip()
        if line.lower().startswith("refined:"):
            refined_text = line.split(":", 1)[1].strip()
        elif line.lower().startswith("score:"):
            score = line.split(":", 1)[1].strip()

    return refined_text, score