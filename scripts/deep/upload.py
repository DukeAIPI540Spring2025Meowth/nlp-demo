from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

def upload_gguf(hf_token, model_path, repo_id):
    api = HfApi(token=hf_token)
    api.create_repo(repo_id, exist_ok=True)
    
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.gguf",  # Your GGUF filename
        repo_id=repo_id
    )
    
def main():
    '''
    Upload the GGUF file to the Hugging Face Hub
    '''
    upload_gguf(
        hf_token=os.getenv("HF_TOKEN"),
        model_path="./conversion_dir/model-q5_k_m.gguf",
        repo_id=f"{os.getenv('HF_USERNAME')}/meowth-nlp-demo-0.1_llama-3.2-3b-instruct_q5_k_m_gguf"
    )

if __name__ == "__main__":
    main()
