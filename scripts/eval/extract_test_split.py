# Load the unified dataset into a vector database for the deep learning approach (RAG)
import pandas as pd
from datasets import load_dataset

def extract_test_split():
    """
    Load the ESConv dataset (test) from Hugging Face.
    
    Returns:
        pandas.DataFrame: Dataset for the HMM advisor
    """
    print("Loading ESConv dataset...")
    dataset = load_dataset("thu-coai/esconv", split="test", trust_remote_code=True)
    data = pd.DataFrame(dataset["test"])
    print(f"Loaded dataset with {len(data)} entries")
    return data
