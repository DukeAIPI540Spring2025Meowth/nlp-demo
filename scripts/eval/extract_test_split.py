import json
from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def extract_test_split():
    '''
    Extract the dataset from the Hugging Face dataset and save it as a JSON Lines file.
    '''
    # Create directory if needed
    Path("esconv").mkdir(exist_ok=True)

    # Load dataset and process entries
    dataset = load_dataset("thu-coai/esconv", split="test", trust_remote_code=True)

    # Save as JSON Lines format
    with open("esconv/test.json", "w", encoding="utf-8") as f:
        for example in dataset:
            try:
                # Parse the JSON string from text field
                parsed = json.loads(example["text"])
                # Write as separate line
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON in example ID {example.get('id', 'unknown')}: {e}")

    print(f"Saved {len(dataset)} lines to esconv/test.json")
