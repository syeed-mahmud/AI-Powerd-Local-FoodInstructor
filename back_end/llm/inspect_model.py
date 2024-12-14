from transformers import AutoModelForCausalLM

def inspect_model(model_id: str):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    # Print model class type
    print(f"Model class: {type(model)}\n")

    # Print the model architecture
    print("Model architecture:")
    print(model)

    # List all submodules
    print("\nSubmodules:")
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    # Replace with your model ID
    model_id = "meta-llama/Llama-3.2-3B"
    inspect_model(model_id)
