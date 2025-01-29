import torch
from dataloader import load_data
from trainer import train_model
from evaluator import evaluate_model

def main():
    """
    Main script to train and evaluate the POS tagger on the full dataset.

    - Loads the full dataset.
    - Trains the model.
    - Evaluates on the test set.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load full dataset
    print("Loading full dataset...")
    train_loader, val_loader, test_loader = load_data()

    # Train the model
    print("Training model...")
    num_tags = 17  # Example: Number of unique POS tags
    model_path = train_model(train_loader, val_loader, num_tags)

    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    evaluate_model(test_loader, num_tags, model_path)

if __name__ == "__main__":
    main()
