import numpy as np
from collections import defaultdict
import re

# Step 1: Split Sentences
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sentences = text.split(".")
    sentences = [sentence.split() for sentence in sentences if sentence]
    return sentences

# Step 2: Make Vocabulary
def build_vocabulary(sentences):
    vocabulary = set()
    for sentence in sentences:
        vocabulary.update(sentence)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

# Step 3: One-Hot Encode
def one_hot_encode(word, word_to_index):
    vector = np.zeros(len(word_to_index))
    vector[word_to_index[word]] = 1
    return vector

# Step 4: Prepare Training Data
def generate_training_data(sentences, word_to_index, window_size=2):
    training_data = []
    for sentence in sentences:
        for i, target_word in enumerate(sentence):
            context = []
            for j in range(-window_size, window_size + 1):
                if j != 0 and 0 <= i + j < len(sentence):
                    context.append(sentence[i + j])
            training_data.append((context, target_word))
    return training_data

# Step 5: Initialize Weights
def initialize_weights(vocab_size, embedding_dim):
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)
    return W1, W2

# Step 6: Forward Pass
def forward_pass(context_words, W1, W2, word_to_index):
    context_vectors = np.sum([one_hot_encode(word, word_to_index) for word in context_words], axis=0)
    hidden_layer = np.dot(context_vectors, W1)
    output_layer = np.dot(hidden_layer, W2)
    predictions = softmax(output_layer)
    return predictions, hidden_layer

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Step 7: Calculate Loss
def calculate_loss(predictions, target_word, word_to_index):
    target_vector = one_hot_encode(target_word, word_to_index)
    loss = -np.sum(target_vector * np.log(predictions))
    return loss

# Step 8: Update Weights
def backpropagate(W1, W2, hidden_layer, context_words, predictions, target_word, word_to_index, learning_rate=0.01):
    target_vector = one_hot_encode(target_word, word_to_index)
    error = predictions - target_vector
    dW2 = np.outer(hidden_layer, error)
    dW1 = np.outer(np.sum([one_hot_encode(word, word_to_index) for word in context_words], axis=0), np.dot(W2, error))
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    return W1, W2

# Example Usage
if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog. The dog barks loudly."
    sentences = preprocess_text(text)
    word_to_index, index_to_word = build_vocabulary(sentences)
    training_data = generate_training_data(sentences, word_to_index)

    vocab_size = len(word_to_index)
    embedding_dim = 10
    W1, W2 = initialize_weights(vocab_size, embedding_dim)

    for epoch in range(100):
        total_loss = 0
        for context_words, target_word in training_data:
            predictions, hidden_layer = forward_pass(context_words, W1, W2, word_to_index)
            loss = calculate_loss(predictions, target_word, word_to_index)
            total_loss += loss
            W1, W2 = backpropagate(W1, W2, hidden_layer, context_words, predictions, target_word, word_to_index)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")