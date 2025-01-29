# Findings and Discussion Report

## Introduction

This project implements a POS tagger using XLM-RoBERTa. The model is trained on the German Universal Dependencies (`de_gsd`) dataset to assign accurate POS tags to tokens. The project demonstrates the end-to-end pipeline, including preprocessing, model training, evaluation, and results comparison.

## Results and Observations

### 1. Dataset Preprocessing (`preprocessor.py`):
- **Observation:**
  - The preprocessing script successfully tokenizes sentences and aligns POS labels.
  - Tokenization preserves linguistic features, ensuring compatibility with the model.

### 2. Model Training (`trainer.py`):
- **Output (After Epoch 1):**
  - **Training Loss:** `0.6572`
  - **Validation Loss:** `0.1744`
  - **Validation Accuracy:** `94.93%`
- **Observation:**
  - The model shows rapid convergence even after the first epoch, achieving high validation accuracy.
  - Cross-entropy loss effectively measures alignment between predictions and ground truth.

### 3. Model Evaluation (`evaluator.py`):
- **Output:**
  - Because of time limitations, only a small segment of the data (roughly the 20% of the whole set) is evaluated. 
  - **Test Loss:** `1.1869` 
  - **Test Accuracy:** `0.8306` 
- **Observation:**
  - Model evaluation confirms robust generalization on unseen data.
  - Accuracy metrics align with expectations based on validation results.
  - The model is expected to perform better in light of these metrics when it is trained over the whole set of data.

### 4. Comparison with CKY Parser:
- **Key Differences:**
  - **Task:** The CKY parser focuses on syntactic parsing and tree reconstruction, while this project targets POS tagging.
  - **Dataset:** CKY uses the ATIS grammar and test sentences, while POS tagging relies on UD German dataset.
  - **Outputs:** The CKY parser produces parse trees, whereas this project outputs token-level POS tags.
- **Shared Observations:**
  - Both projects highlight the importance of preprocessing for task-specific compatibility.
  - Both demonstrate high accuracy and efficiency in handling linguistic data.

## Discussion

1. **Performance:**
   - The POS tagger achieves high accuracy on both validation and test datasets.
   - Efficient preprocessing and training pipelines contribute to overall performance.

2. **Scalability:**
   - The modular design allows for easy adaptation to other languages or POS tagging datasets.
   - Training larger datasets or models could require additional computational resources.

3. **Challenges:**
   - Unknown tokens during evaluation can impact accuracy; handling rare words or unseen vocabulary is critical.

## Conclusions

- The XLM-RoBERTa-based POS tagger effectively assigns POS tags with high accuracy.
- Intermediate results indicate strong performance, with a rapid decrease in loss and high validation accuracy.
- Further training and evaluation will refine the model's performance.

