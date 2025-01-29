from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast
import torch

def preprocess():
    """
    Preprocess the dataset for POS tagging.

    - Load the dataset and tokenizer.
    - Map POS tags to IDs.
    - Tokenize and align labels.
    - Save the processed dataset.

    :return: None
    """
    data = load_dataset("universal_dependencies", "de_gsd", trust_remote_code=True)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    tags = {tag for sample in data["train"] for tag in sample["upos"]}
    tag_to_id = {tag: i for i, tag in enumerate(sorted(tags))}

    print(f"Tag mappings: {tag_to_id}")

    def align_labels(samples):
        """
        Tokenize inputs and align POS labels.

        :param samples: Batch of sentences and their POS tags.
        :return: Tokenized sentences with aligned labels.
        """
        tokens = tokenizer(
            samples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True
        )

        labels = []
        for i, sample_tags in enumerate(samples["upos"]):
            word_ids = tokens.word_ids(batch_index=i)
            tag_ids = []
            prev_word = None

            for word in word_ids:
                if word is None:
                    tag_ids.append(-100)
                elif word != prev_word:
                    tag_ids.append(tag_to_id.get(sample_tags[word], -100))
                else:
                    tag_ids.append(-100)
                prev_word = word

            labels.append(tag_ids)

        tokens["labels"] = labels
        return tokens

    processed_data = data.map(align_labels, batched=True)

    output_path = "tokenized_data.pth"
    torch.save(processed_data, output_path)

if __name__ == "__main__":
    preprocess()
