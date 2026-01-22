from model_wrapper import ModelWrapper

# Example usage for interpretability research
def main():
    # Initialize the model wrapper
    model = ModelWrapper(
        model_name="microsoft/DialoGPT-medium",  # Replace with your preferred model
        device="cpu",  # or "cpu"
        torch_dtype=None,  # Will default to float16
        trust_remote_code=False
    )
    
    # Print model information
    print("Model Info:")
    print(model.get_model_info())
    print()
    
    # Example 1: Simple conversation
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = model.prompt_with_chat_template(
        messages=messages,
        max_new_tokens=100,
        temperature=0.7
    )
    
    print("Response 1:")
    print(response)
    print()
    
    # Example 2: Multi-turn conversation with system prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in science."},
        {"role": "user", "content": "Explain quantum entanglement in simple terms."},
        {"role": "assistant", "content": "Quantum entanglement is when two particles become connected in such a way that measuring one instantly affects the other, no matter how far apart they are."},
        {"role": "user", "content": "How is this used in quantum computing?"}
    ]
    
    response = model.prompt_with_chat_template(
        messages=messages,
        max_new_tokens=150,
        temperature=0.5,
        top_p=0.9
    )
    
    print("Response 2:")
    print(response)
    print()
    
    # Example 3: For interpretability research - controlled generation
    interpretability_messages = [
        {"role": "user", "content": "Complete this sentence: The weather today is"}
    ]
    
    # Generate multiple responses for analysis
    print("Multiple responses for interpretability analysis:")
    for i in range(3):
        response = model.prompt_with_chat_template(
            messages=interpretability_messages,
            max_new_tokens=20,
            temperature=0.8,
            do_sample=True
        )
        print(f"Response {i+1}: {response}")

if __name__ == "__main__":
    main()