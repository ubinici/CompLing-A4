import torch
from transformers import XLMRobertaTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader

def load_data(batch_size=16):
    """
    Load and prepare data loaders for training, validation, and testing.

    :param batch_size: Batch size for data loaders.
    :return: Tuple of (train_loader, val_loader, test_loader).
    """
    data = torch.load("tokenized_data.pth", weights_only=False)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    def clean(batch):
        """
        Retain only necessary fields for training.

        :param batch: A single batch of data.
        :return: Cleaned batch with input_ids, attention_mask, and labels.
        """
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    data = {split: dataset.map(clean, remove_columns=dataset.column_names)
            for split, dataset in data.items()}

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(data["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(data["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(data["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()
    for batch in train_loader:
        print("Sample batch keys:", batch.keys())
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Attention mask shape:", batch["attention_mask"].shape)
        print("Labels shape:", batch["labels"].shape)
        break
