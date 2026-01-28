import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. Reproducibility ---
def reset_seeds():
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED']='0'
    np.random.seed(42)
    tf.random.set_seed(42)

reset_seeds()

# --- 2. Synthetic Data Generation ---
def generate_synthetic_data(num_samples=1000, seq_length=150):
    """
    Generates sequences where the label depends on long-term memory.
    Label is 1 if the count of numbers > 50 in the first window (5-25)
    is greater than the count in the second window (130-150).
    """
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.randint(0, 100, size=seq_length)
        
        # Window 1 & 2
        window1 = np.random.randint(0, 100, size=20)
        seq[5:25] = window1
        count1 = np.sum(window1 > 50)
        
        noise = np.random.randint(0, 100, size=105)
        seq[25:130] = noise
        
        window2 = np.random.randint(0, 100, size=20)
        seq[130:150] = window2
        count2 = np.sum(window2 > 50)
        
        label = 1 if count1 > count2 else 0
        X.append(seq)
        y.append(label)
        
    return np.array(X), np.array(y)

# --- 3. Model Definitions ---
SEQ_LENGTH = 150
VOCAB_SIZE = 100
EMBEDDING_DIM = 16

def build_dense_model():
    """Builds a standard Dense (MLP) model that flattens the sequence."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model():
    """Builds a SimpleRNN model."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.SimpleRNN(64, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model():
    """Builds an LSTM model."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.LSTM(64, return_sequences=True, activation='tanh'),
        layers.LSTM(32, activation='tanh'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 4. Plotting Function ---
def plot_history(histories, names):
    """Plots the validation accuracy for a list of model histories."""
    plt.figure(figsize=(12, 7))
    for hist, name in zip(histories, names):
        val_acc = hist.history['val_accuracy']
        plt.plot(val_acc, label=f'{name} Val Acc (Final: {val_acc[-1]:.3f})')
    plt.title('Model Comparison: Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 5. Main Execution ---
def main():
    """Main function to run the experiment."""
    print("="*60)
    print("CASE STUDY: SEQUENTIAL DATA IN NLP")
    print("="*60)

    # Generate and split data
    reset_seeds()
    X, y = generate_synthetic_data(1000, SEQ_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
    
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    histories = []
    model_names = ['Dense', 'RNN', 'LSTM']
    model_builders = [build_dense_model, build_rnn_model, build_lstm_model]

    for name, builder in zip(model_names, model_builders):
        print(f"--- Training {name} Model ---")
        reset_seeds()
        model = builder()
        model.summary()
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        histories.append(history)
        print("-"*(20 + len(name)) + "\n")

    # Plot results
    plot_history(histories, model_names)

if __name__ == '__main__':
    main()
