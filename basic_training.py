"""
Basic training example for SLM-Trainer.

This script demonstrates how to:
1. Train a small language model on text data
2. Save the trained model
3. Load the model
4. Generate text
"""

from slm_trainer import SLMTrainer


def main():
    print("=" * 70)
    print("SLM-Trainer: Basic Training Example")
    print("=" * 70)

    # Step 1: Create sample training data
    print("\nStep 1: Creating sample training data...")

    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Artificial intelligence is transforming the world.
    Machine learning models can learn from data.
    Natural language processing enables computers to understand human language.
    Deep learning uses neural networks with multiple layers.
    Transformers are a type of neural network architecture.
    GPT models are based on the transformer architecture.
    Language models can generate human-like text.
    Training data is crucial for machine learning.
    """ * 100  # Repeat to create more training data

    # Write to file
    with open("sample_data.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    print("Sample data created: sample_data.txt")

    # Step 2: Initialize trainer
    print("\nStep 2: Initializing SLM-Trainer...")
    trainer = SLMTrainer(
        model_size="tiny",  # Use tiny model for quick training
        vocab_size=512  # Small vocab for limited training data
    )

    # Step 3: Train the model
    print("\nStep 3: Training the model...")
    trainer.train(
        data="sample_data.txt",
        epochs=5,
        batch_size=8
    )

    # Step 4: Save the model
    print("\nStep 4: Saving the model...")
    trainer.save("./my_first_slm")
    print("Model saved to: ./my_first_slm")

    # Step 5: Load the model
    print("\nStep 5: Loading the model...")
    loaded_trainer = SLMTrainer.load("./my_first_slm")
    print("Model loaded successfully!")

    # Step 6: Generate text
    print("\nStep 6: Generating text...")
    prompts = [
        "The quick brown fox",
        "Artificial intelligence",
        "Machine learning",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated_text = loaded_trainer.generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            top_p=0.9
        )
        print(f"Generated: {generated_text}")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
