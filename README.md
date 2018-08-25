"# Automatic-Questions-Generator" 
Install requirements:
  pip install -r requirements.txt.
Data-set:
  - SQuAD - New reading comprehension data-set
  - Contains 100k examples
  - Structure – Paragraph , question, answer
  - Split to 70% train and 30% test
Words Embedding:
  - GloVe – Unsupervised learning algorithm for obtaining vector representations for words
  - Preserve the meaning of the words
Model Architecture – Sequence to Sequence:
  Name Entity Recognition - NLP tools that extract potential answers from paragraphs
  Encoder- Encode the input answers into the attention vector
  Encoder hidden layer - Recurrent Neural Network that capable of learning long-term dependencies
  Attention mechanism - Gives the neural network the ability to "focus" on part of the input
  Decoder - Generate the output question word by word using the attention vector
  Dense layer - Turn the top hidden states to logit vectors
  Softmax layer - Provide probabilities for each possible class
  My bonus layer - Increase the probability for classes in the current input
