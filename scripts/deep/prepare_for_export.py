from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_dir = "./tuned_model/epoch_4"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.save_pretrained("conversion_dir")
    tokenizer.save_pretrained("conversion_dir")

if __name__ == "__main__":
    main()
