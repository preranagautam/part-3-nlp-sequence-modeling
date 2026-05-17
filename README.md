# part-3-nlp-sequence-modeling

# NLP and Sequence Modeling Mini Project

### Overview

This project demonstrates a complete Natural Language Processing (NLP) pipeline for customer support sentiment classification using both traditional machine learning and deep learning sequence modeling approaches.

The project focuses on understanding:
- text preprocessing
- text vectorization
- baseline NLP models
- sequence modeling using LSTM
- attention and transformer concepts

The goal is to classify customer support messages into:
- Positive
- Neutral
- Negative

sentiment categories.

---

## Dataset

Dataset Source:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

The original dataset files were not modified.

---

## Dataset Information

The dataset contains customer support interactions with the following columns:

| Column Name | Description |
|---|---|
| ticket_id | Unique ticket identifier |
| channel | Communication channel |
| customer_message | Customer support message text |
| sentiment_label | Sentiment class label |
| word_count | Number of words in the message |
| urgent_flag | Indicates urgency of the ticket |

For this assignment, the primary NLP task focused on:
- Input Text → `customer_message`
- Target Label → `sentiment_label`

Additional metadata columns were retained for analysis purposes but were not used in the core NLP modeling pipeline.

---

## Project Structure

```text
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
│
├── results/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.csv
│   └── metrics.json
│
└── sequence_model_architecture.md
```

---

## Task 1 — Dataset Understanding

The dataset was analyzed to understand:
- total number of records
- target class distribution
- text characteristics
- average text length
- sample customer messages

### Dataset Shape

```python
print(df.shape)
```

### Class Distribution

```python
print(df['sentiment_label'].value_counts())
```

### Average Text Length

The average text length was calculated using word counts from customer messages.

### Visualizations Generated

- Sentiment class distribution plot
- Text length distribution histogram

---

## Task 2 — Text Preprocessing

A preprocessing pipeline was implemented to clean and standardize textual data before modeling.

### Preprocessing Steps

#### 1. Lowercasing

All text was converted to lowercase to maintain consistency.

Example:

```text
"Payment Failed"
↓
"payment failed"
```

---

#### 2. Removing Special Characters

Special symbols, punctuation, and unnecessary characters were removed using regular expressions.

---

#### 3. Tokenization

Text was split into individual words (tokens).

Example:

```text
"refund still pending"
↓
["refund", "still", "pending"]
```

---

#### 4. Stopword Removal

Common English stopwords such as:
- the
- is
- and
- was

were removed to reduce noise.

---

#### 5. Sequence Padding

For sequence models such as LSTM, input sequences were padded to fixed lengths.

Example:

```text
[15, 42, 8]
↓
[15, 42, 8, 0, 0]
```

Padding ensures all sequences have equal dimensions during model training.

---

## Task 3 — Text Vectorization

Machine learning models cannot process raw text directly. Therefore, text must be converted into numerical representations.

### Why Text Must Be Converted into Vectors

Models operate using mathematical computations and numerical matrices.

Text vectorization converts language into numerical form while preserving:
- word frequency
- contextual information
- semantic relationships

---

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) was used for the baseline model.

#### Advantages

- lightweight
- fast
- interpretable
- effective for traditional NLP models

Implementation:

```python
TfidfVectorizer(max_features=5000)
```

---

### Tokenizer-Based Sequences

For sequence modeling, text was converted into token sequences using Keras Tokenizer.

Example:

```text
"I need refund"
↓
[14, 85, 201]
```

These sequences were then padded before being passed into the LSTM network.

---

## Task 4 — Baseline Model

A traditional machine learning baseline model was implemented using:

- TF-IDF Vectorization
- Logistic Regression

---

### Baseline Model Workflow

```text
Customer Message
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Sentiment Prediction
```

---

### Evaluation Metrics

The baseline model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Task 5 — Sequence Modeling using LSTM

A Long Short-Term Memory (LSTM) based sequence model was implemented.

Unlike traditional vectorization methods, LSTMs preserve:
- word order
- contextual meaning
- sequential dependencies

---

### Sequence Model Architecture

```text
Customer Message
        ↓
Text Cleaning
        ↓
Tokenization
        ↓
Sequence Padding
        ↓
Embedding Layer
        ↓
LSTM Layer
        ↓
Dense Output Layer
        ↓
Sentiment Prediction
```

---

### Input Sequence

Customer messages were converted into integer token sequences.

Example:

```text
"My refund is delayed"
↓
[15, 42, 8, 91]
```

Sequences were padded to fixed length before training.

---

### Embedding Layer

The embedding layer transforms token IDs into dense vector representations.

Implementation:

```python
Embedding(
    input_dim=10000,
    output_dim=128
)
```

This layer helps the model learn semantic relationships between words.

---

### Recurrent / Sequence Layer

An LSTM layer was used to process sequential text data.

Implementation:

```python
LSTM(
    64,
    dropout=0.2,
    recurrent_dropout=0.2
)
```

The LSTM captures:
- contextual dependencies
- sequence order
- long-term relationships between words

---

### Output Layer

The final dense layer predicts sentiment probabilities using softmax activation.

Implementation:

```python
Dense(
    3,
    activation='softmax'
)
```

The three output classes are:
- positive
- neutral
- negative

---

### Loss Function

The sequence model used:

```text
Sparse Categorical Crossentropy
```

This loss function is suitable for:
- multi-class classification
- integer encoded labels

---

### Evaluation Metric

Primary metric used:

```text
Accuracy
```

Additional evaluation metrics:
- Precision
- Recall
- F1-score

---

## Task 6 — Attention and Transformer Reflection

### Why RNNs Struggle with Long-Term Dependencies

Traditional RNNs process sequences step-by-step and often suffer from:
- vanishing gradients
- exploding gradients

As sequence length increases, earlier information may be forgotten.

---

### How LSTMs Help with Memory

LSTMs introduce:
- forget gates
- input gates
- output gates
- cell states

These mechanisms allow important information to persist across long sequences.

This improves contextual understanding and reduces memory loss.

---

### What Attention Solves

Attention mechanisms allow models to focus on the most relevant parts of an input sequence during prediction.

Instead of compressing all information into a single hidden state, attention dynamically assigns importance to different words.

This significantly improves:
- translation quality
- contextual understanding
- long-sequence processing

---

### Why Transformers Are Important

Transformers rely entirely on self-attention instead of recurrence.

Advantages include:
- parallel processing
- better scalability
- improved contextual learning
- efficient long-sequence handling

Modern NLP systems such as:
- BERT
- GPT
- T5
- LLaMA

are all transformer-based architectures.

Transformers form the foundation of modern Generative AI systems.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- NLTK

---

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Open Jupyter Notebook

```bash
jupyter notebook
```

---

### Step 2 — Run Notebook

Run all cells in:

```text
notebook.ipynb
```

---

## Results Summary

| Model | Representation | Objective |
|---|---|---|
| Logistic Regression | TF-IDF | Baseline NLP Classification |
| LSTM | Token Sequences | Sequence Modeling |

---

## Generated Outputs

The project generates:
- class distribution plots
- confusion matrix visualizations
- sample prediction files
- evaluation metrics

These outputs are stored in the `results/` directory.

---

## Key Learnings

- Text data must be converted into numerical vectors before modeling.
- TF-IDF provides strong traditional NLP baselines.
- Sequence models preserve contextual word relationships.
- LSTMs improve memory handling compared to vanilla RNNs.
- Transformers dominate modern NLP because of attention mechanisms.

---

## Future Improvements

Potential enhancements:
- pretrained embeddings (Word2Vec, GloVe)
- Bidirectional LSTM
- Attention layers
- Transformer fine-tuning
- Hyperparameter optimization
- BERT-based sentiment classification

---

## Conclusion

This project successfully implemented:
- text preprocessing
- text vectorization
- traditional NLP classification
- sequence modeling using LSTM
- theoretical understanding of modern NLP architectures

The project demonstrates both practical implementation and conceptual understanding of sequence-based Natural Language Processing systems.

---

## Author

Prerana Gautam Shukla
