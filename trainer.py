import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from model import POSModel
from dataloader import load_data


def train_model(train_loader, val_loader, num_tags, epochs=5, lr=2e-5):
    """
    Train the POS tagging model.

    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param num_tags: Number of unique POS tags.
    :param epochs: Number of training epochs.
    :param lr: Learning rate.
    :return: Path to the saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss, optimizer, and scheduler
    model = POSModel(num_tags).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, num_tags), labels.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

        # Validation
        validate_model(model, val_loader, loss_fn, device)

    # Save the trained model
    model_path = "pos_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model_path


def validate_model(model, val_loader, loss_fn, device):
    """
    Validate the model on the validation set.

    :param model: Trained model.
    :param val_loader: DataLoader for the validation set.
    :param loss_fn: Loss function.
    :param device: Device to use.
    """
    model.eval()
    val_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            val_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_loader, val_loader, _ = load_data()
    num_tags = 17  # Example: Number of unique POS tags
    train_model(train_loader, val_loader, num_tags)

