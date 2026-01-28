
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# =========================================
# REPRODUCIBILITY
# =========================================
def set_seeds():
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    tf.random.set_seed(42)

# =========================================
# PART 1: DATA SYNTHESIS
# =========================================
def generate_synthetic_data(num_samples=1000, seq_length=150):
    """
    Generates sequences with a true sequential memory requirement.
    Label is 1 if the count of numbers > 50 in the first window (5-25)
    is greater than the count in the second window (130-150).
    """
    X = []
    y = []
    for _ in range(num_samples):
        seq = np.random.randint(0, 100, size=seq_length)
        
        # Window 1: positions 5-25
        window1 = np.random.randint(0, 100, size=20)
        seq[5:25] = window1
        count1 = np.sum(window1 > 50)
        
        # Noise: positions 25-130
        noise = np.random.randint(0, 100, size=105)
        seq[25:130] = noise
        
        # Window 2: positions 130-150
        window2 = np.random.randint(0, 100, size=20)
        seq[130:150] = window2
        count2 = np.sum(window2 > 50)
        
        # Label: 1 if Window 1 has MORE values > 50
        label = 1 if count1 > count2 else 0
        
        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)

# =========================================
# PART 2: MODEL DEFINITIONS
# =========================================

# --- Hyperparameters ---
SEQ_LENGTH = 150
VOCAB_SIZE = 100
EMBEDDING_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20

def build_dense_model():
    """Builds, compiles, and returns the Dense model."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_rnn_model():
    """Builds, compiles, and returns the Simple RNN model."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.SimpleRNN(32, activation='tanh'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model():
    """Builds, compiles, and returns the LSTM model."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LENGTH,)),
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        layers.LSTM(32, activation='tanh'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================================
# PART 3: TRAINING & EVALUATION
# =========================================

def train_and_evaluate(run_number=1):
    """
    Main function to run the full pipeline:
    1. Set seeds
    2. Generate data
    3. Build, train, and evaluate all three models
    4. Print a report of the results.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING VALIDATION PIPELINE: RUN #{run_number}")
    print(f"{'='*60}\n")

    set_seeds()

    # --- 1. Data Loading and Preprocessing ---
    print("[STEP 1/4] Generating and splitting synthetic data...")
    X, y = generate_synthetic_data(1000, SEQ_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
    print(f"Data generated. Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    models_to_train = {
        "Dense": build_dense_model(),
        "RNN": build_rnn_model(),
        "LSTM": build_lstm_model(),
    }

    results = {}

    for i, (name, model) in enumerate(models_to_train.items()):
        print(f"[STEP {i+2}/4] Training {name} model...")
        model.fit(X_train, y_train, 
                  epochs=EPOCHS, 
                  batch_size=BATCH_SIZE, 
                  validation_data=(X_test, y_test), 
                  verbose=1)
        
        # --- Evaluation ---
        print(f"Evaluating {name} model...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'loss': loss, 'accuracy': accuracy, 'f1_score': f1}
        print(f"{name} Results -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}\n")

    # --- 4. Final Report for the run ---
    print(f"\n--- RESULTS FOR RUN #{run_number} ---")
    for name, metrics in results.items():
        print(f"  - {name}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")
    print("---------------------------\n")
    
    return results

# =========================================
# MAIN EXECUTION
# =========================================
if __name__ == "__main__":
    NUM_VALIDATION_RUNS = 2
    all_run_results = []

    for i in range(NUM_VALIDATION_RUNS):
        run_results = train_and_evaluate(run_number=i+1)
        all_run_results.append(run_results)

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE SUMMARY ACROSS ALL RUNS")
    print(f"{'='*60}\n")

    # Calculate and print average metrics
    avg_results = {}
    for run_result in all_run_results:
        for model_name, metrics in run_result.items():
            if model_name not in avg_results:
                avg_results[model_name] = {'accuracy': [], 'f1_score': []}
            avg_results[model_name]['accuracy'].append(metrics['accuracy'])
            avg_results[model_name]['f1_score'].append(metrics['f1_score'])
            
    print("Average Performance:")
    for model_name, metrics_lists in avg_results.items():
        avg_acc = np.mean(metrics_lists['accuracy'])
        avg_f1 = np.mean(metrics_lists['f1_score'])
        print(f"  - {model_name}: Avg Accuracy={avg_acc:.4f}, Avg F1-Score={avg_f1:.4f}")

    print("\nObjective: LSTM > RNN > Dense")
    print("SUCCESS CRITERIA MET: To be determined by the final average scores.")
