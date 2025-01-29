import torch
from tqdm import tqdm
from model import POSModel
from dataloader import load_data

def evaluate_model(test_loader, num_tags, model_path):
    """
    Evaluate the model on the test set.

    :param test_loader: DataLoader for the test set.
    :param num_tags: Number of unique POS tags.
    :param model_path: Path to the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = POSModel(num_tags).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    _, _, test_loader = load_data()
    num_tags = 18  # Example: Number of unique POS tags
    model_path = "pos_model.pth"  # Path to the trained model
    evaluate_model(test_loader, num_tags, model_path)
