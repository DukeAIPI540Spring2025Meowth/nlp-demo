from llama_cpp import Llama
from termcolor import colored

def main():
    llm = Llama(
        model_path="conversion_dir/model-q5_k_m.gguf",
        n_ctx=2048, 
        n_threads=8,
        n_gpu_layers=0,  # Disable Metal GPU acceleration
        offload_kqv=True  # Optimize CPU memory usage
    )

    system_message = "You are a helpful mental health assistant."
    user_prompt = "I haven't been feeling like myself lately."
    
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.7,
        max_tokens=100,
        stop=["</s>"],
    )
    print('\n' + colored(response['choices'][0]['message']['content'], 'green') + '\n')

if __name__ == "__main__":
    main()
