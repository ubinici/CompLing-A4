import torch
from torch import nn
from transformers import XLMRobertaModel

class POSModel(nn.Module):
    """
    POS Tagger model based on XLM-RoBERTa.

    - Uses pre-trained XLM-RoBERTa as the base.
    - Adds a classification head for POS tagging.
    - Freezes embeddings and a subset of encoder layers for efficiency.
    """
    def __init__(self, num_tags):
        """
        Initialize the POSModel.

        :param num_tags: Number of unique POS tags.
        """
        super(POSModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        # Freeze embeddings and first 9 encoder layers
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for param in self.roberta.encoder.layer[:9].parameters():
            param.requires_grad = False

        # Classification head
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model.

        :param input_ids: Input token IDs.
        :param attention_mask: Attention mask for the input.
        :return: Logits for each token.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)
        return logits

if __name__ == "__main__":
    # Parameters
    num_tags = 17  # Example number of POS tags
    batch_size = 16
    seq_length = 128

    # Dummy inputs for testing
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Initialize model
    model = POSModel(num_tags)

    # Forward pass
    logits = model(input_ids, attention_mask)

    # Output shapes
    print(f"Logits shape: {logits.shape}")
