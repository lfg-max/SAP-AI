
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

# =========================================
# PART 1: REPRODUCIBILITY
# =========================================
def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds set with value: {seed}")

# =========================================
# PART 2: DATA SYNTHESIS
# =========================================
def generate_synthetic_data(num_samples=1000, seq_length=150):
    """
    Generates sequences where the label depends on long-range dependencies.
    Rule: Label is 1 if the count of numbers > 50 in the first window (5-25)
          is greater than the count in the second window (130-150).
    """
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.randint(0, 100, size=seq_length)
        
        # Window 1
        window1 = np.random.randint(0, 100, size=20)
        seq[5:25] = window1
        count1 = np.sum(window1 > 50)
        
        # Noise
        noise = np.random.randint(0, 100, size=105)
        seq[25:130] = noise
        
        # Window 2
        window2 = np.random.randint(0, 100, size=20)
        seq[130:150] = window2
        count2 = np.sum(window2 > 50)
        
        label = 1 if count1 > count2 else 0
        X.append(seq)
        y.append(label)
        
    return np.array(X), np.array(y)

# =========================================
# PART 3: MODEL DEFINITIONS
# =========================================
def build_dense_model(seq_length, vocab_size, embedding_dim):
    """Builds a standard Dense model."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model(seq_length, vocab_size, embedding_dim):
    """Builds a SimpleRNN model."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.SimpleRNN(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(seq_length, vocab_size, embedding_dim):
    """Builds an LSTM model."""
    model = models.Sequential([
        layers.Input(shape=(seq_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# =========================================
# PART 4: TRAINING & EVALUATION
# =========================================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, epochs=20, batch_size=32):
    """Trains and evaluates a single model."""
    print(f"--- Training {model_name} Model ---")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluation
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{model_name} Performance: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}, Loss={loss:.4f}")
    return {"model": model_name, "accuracy": accuracy, "f1_score": f1, "loss": loss}

# =========================================
# PART 5: MAIN EXECUTION
# =========================================
def main():
    """Main function to run the experiment."""
    # Hyperparameters
    SEQ_LENGTH = 150
    VOCAB_SIZE = 100
    EMBEDDING_DIM = 12
    NUM_SAMPLES = 2000
    TEST_SIZE = 0.3
    EPOCHS = 30
    BATCH_SIZE = 64
    
    # Run the experiment multiple times for verification
    num_runs = 3
    results_over_runs = []
    
    for i in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{num_runs}")
        print(f"{'='*60}\n")
        
        # Set seeds for this run
        run_seed = 42 + i
        set_seeds(run_seed)
        
        # 1. Generate Data
        print("Step 1: Generating synthetic data...")
        X, y = generate_synthetic_data(num_samples=NUM_SAMPLES, seq_length=SEQ_LENGTH)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=run_seed)
        print(f"Data generated: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")

        # 2. Build Models
        print("\nStep 2: Building models...")
        dense_model = build_dense_model(SEQ_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)
        rnn_model = build_rnn_model(SEQ_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)
        lstm_model = build_lstm_model(SEQ_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)
        print("Models built successfully.")

        # 3. Train and Evaluate
        print("\nStep 3: Training and evaluating models...")
        run_results = []
        
        # Dense Model
        results = train_and_evaluate(dense_model, X_train, y_train, X_test, y_test, "Dense", epochs=EPOCHS, batch_size=BATCH_SIZE)
        run_results.append(results)
        
        # RNN Model
        results = train_and_evaluate(rnn_model, X_train, y_train, X_test, y_test, "RNN", epochs=EPOCHS, batch_size=BATCH_SIZE)
        run_results.append(results)
        
        # LSTM Model
        results = train_and_evaluate(lstm_model, X_train, y_train, X_test, y_test, "LSTM", epochs=EPOCHS, batch_size=BATCH_SIZE)
        run_results.append(results)
        
        results_over_runs.append(run_results)

    # 4. Final Report
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE REPORT")
    print(f"{'='*60}\n")

    for i, run_results in enumerate(results_over_runs):
        print(f"--- Results for Run {i+1} ---")
        for res in run_results:
            print(f"  - {res['model']}: Accuracy={res['accuracy']:.4f}, F1-Score={res['f1_score']:.4f}")
        print("\n")

if __name__ == '__main__':
    main()
