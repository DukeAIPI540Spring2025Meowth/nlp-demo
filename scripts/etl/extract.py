# Extract datasets from sources
# Dataset link: https://huggingface.co/datasets/thu-coai/esconv?library=datasets
import pandas as pd
from datasets import load_dataset

def load_data():
    """
    Load the ESConv dataset from Hugging Face.
    
    Returns:
        pandas.DataFrame: Dataset for the HMM advisor
    """
    print("Loading ESConv dataset...")
    dataset = load_dataset("thu-coai/esconv")
    data = pd.DataFrame(dataset["train"])
    print(f"Loaded dataset with {len(data)} entries")
    return data

