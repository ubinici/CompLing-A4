import torch
from model import POSModel
from dataloader import load_data
from trainer import train_model
from evaluator import evaluate_model
import os

def run_demo():
    """
    Demonstration script to train and evaluate a POS tagger on a smaller dataset.

    - Trains on a subset of the training data.
    - Evaluates on a subset of the validation data.
    - Saves evaluation results to a text file.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, _ = load_data()

    # Use smaller datasets for demonstration
    print("Preparing smaller datasets for demo...")
    small_train_size = 500
    small_val_size = 100

    small_train_loader = torch.utils.data.DataLoader(
        list(train_loader.dataset)[:small_train_size],
        batch_size=16,
        shuffle=True,
        collate_fn=train_loader.collate_fn,
    )

    small_val_loader = torch.utils.data.DataLoader(
        list(val_loader.dataset)[:small_val_size],
        batch_size=16,
        shuffle=False,
        collate_fn=val_loader.collate_fn,
    )

    # Train the model
    print("Training on a small dataset...")
    num_tags = 18  # Example: Number of unique POS tags
    model_path = train_model(small_train_loader, small_val_loader, num_tags, epochs=2)

    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(small_val_loader, num_tags, model_path)

    # Save results to a file
    results_file = "demo_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Demo completed successfully. Model trained and evaluated on small dataset.\n")
    print(f"Results saved to {results_file}.")

if __name__ == "__main__":
    run_demo()
