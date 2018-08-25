# Automatic-Questions-Generator

The rate of digital texts creation is increasing. While some of them contain comprehension questions, which allows the readers to verify understanding, most of them do not.
This project’s purpose is to develop an automatic algorithm that generate questions, which is intended to help both text writers and readers.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python3
* Install requirements packages using the requirements.txt file

```
pip install -r requirements.txt
```

### Data-set

* **SQuAD** - New reading comprehension data-set
* Contains 100k examples
* Structure – Paragraph , question, answer
* Split to 70% train and 30% test

### Words Embedding
* **GloVe** – Unsupervised learning algorithm for obtaining vector representations for words
* Preserve the meaning of the words

### Model Architecture – Sequence to Sequence

* **Name Entity Recognition** - NLP tools that extract potential answers from paragraphs
* **Encoder** - Encode the input answers into the attention vector
* **Encoder hidden layer** - Recurrent Neural Network that capable of learning long-term dependencies
* **Attention mechanism** - Gives the neural network the ability to "focus" on part of the input
* **Decoder** - Generate the output question word by word using the attention vector
* **Dense layer** - Turn the top hidden states to logit vectors
* **Softmax layer** - Provide probabilities for each possible class
* **My bonus layer** - Increase the probability for classes in the current input

### Evaluation methods
* **BLEU** – Score for comparing a candidate generation of text to one or more reference texts
* **F1 score** – Weighted harmonic mean of Precision and Recall
