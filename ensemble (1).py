from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D,
                                     Concatenate, Dense, Dropout, SpatialDropout1D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
import tensorflow as tf
import evaluate
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B prompts


# ----------------------------
# BERT-related imports
# ----------------------------

# ----------------------------
# CNN-related imports (TensorFlow/Keras)
# ----------------------------


def train_bert(train_df, test_df):
    """
    Train the BERT model using Hugging Face Trainer and return test probabilities.
    """
    # Split training data (80/20 train/val)
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["target"]
    )
    print("BERT - Number of training samples:", len(train_split_df))
    print("BERT - Number of validation samples:", len(val_split_df))

    # Convert DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_split_df)
    val_dataset = Dataset.from_pandas(val_split_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and define tokenization function
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Rename column so Trainer can compute loss
    train_dataset = train_dataset.rename_column("target", "labels")
    val_dataset = val_dataset.rename_column("target", "labels")

    # Set format for PyTorch tensors
    train_dataset.set_format(type="torch", columns=[
                             "input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=[
                           "input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=[
                            "input_ids", "attention_mask"])

    # Load pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    # Define F1 metric using evaluate
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return f1_metric.compute(predictions=preds, references=labels, average="weighted")

    # Define training arguments with reduced batch size for memory constraints
    training_args = TrainingArguments(
        output_dir="./results_bert",
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced from 16 to 8
        per_device_eval_batch_size=8,   # Reduced from 16 to 8
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=["none"],
        disable_tqdm=True,
        seed=42
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train BERT
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print("BERT Validation Results:", eval_results)

    # Predict on test set
    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    # Compute probabilities using softmax and select probability for class 1
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    bert_test_probs = probs[:, 1]
    return bert_test_probs, trainer, tokenizer


def train_cnn(train_df, test_df):
    """
    Train the CNN model using TensorFlow/Keras and return test probabilities.
    """
    # Extract texts and labels
    texts_train = train_df['text'].astype(str).values
    labels_train = train_df['target'].values
    texts_test = test_df['text'].astype(str).values

    # Parameters for tokenization and model
    max_words = 20000    # Maximum number of words to consider
    max_length = 100     # Maximum sequence length (in tokens)
    embedding_dim = 128  # Embedding dimension

    # Create and fit the Keras tokenizer
    keras_tokenizer = KerasTokenizer(num_words=max_words)
    keras_tokenizer.fit_on_texts(texts_train)

    # Convert texts to sequences and pad them
    train_sequences = keras_tokenizer.texts_to_sequences(texts_train)
    test_sequences = keras_tokenizer.texts_to_sequences(texts_test)
    X_train_full = pad_sequences(train_sequences, maxlen=max_length)
    X_test = pad_sequences(test_sequences, maxlen=max_length)

    # Custom train/validation split function
    def custom_train_test_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split = int(X.shape[0] * test_size)
        test_idx = indices[:split]
        train_idx = indices[split:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    X_train, X_val, y_train, y_val = custom_train_test_split(
        X_train_full, labels_train, test_size=0.2, random_state=42)
    print("CNN - Number of training samples:", X_train.shape[0])
    print("CNN - Number of validation samples:", X_val.shape[0])

    # Build the multi-kernel CNN model
    input_layer = Input(shape=(max_length,), name='input')
    embedding_layer = Embedding(input_dim=max_words,
                                output_dim=embedding_dim,
                                input_length=max_length,
                                name='embedding')(input_layer)
    x = SpatialDropout1D(0.2)(embedding_layer)

    # Create parallel convolutional layers with different kernel sizes
    filter_sizes = [3, 4, 5]
    conv_layers = []
    for size in filter_sizes:
        conv = Conv1D(filters=128, kernel_size=size, activation='relu')(x)
        pool = GlobalMaxPooling1D()(conv)
        conv_layers.append(pool)

    concatenated = Concatenate()(conv_layers)
    x = Dropout(0.5)(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    cnn_model = Model(inputs=input_layer, outputs=output_layer)
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    cnn_model.summary()

    # Train the CNN model with early stopping
    batch_size = 32
    epochs = 20
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)

    history = cnn_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # (Optional) Evaluate on the validation set
    y_val_pred_prob = cnn_model.predict(X_val)
    y_val_pred = (y_val_pred_prob > 0.5).astype(int)

    def custom_f1_score(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    f1 = custom_f1_score(np.array(y_val), y_val_pred)
    print("CNN Validation F1 Score: {:.4f}".format(f1))

    # Predict on the test set (predictions are probabilities)
    cnn_test_probs = cnn_model.predict(X_test).flatten()
    return cnn_test_probs, cnn_model, keras_tokenizer


def ensemble_predictions(bert_probs, cnn_probs, threshold=0.5):
    """
    Ensemble by averaging probabilities from both models and thresholding.
    """
    ensemble_probs = (bert_probs + cnn_probs) / 2.0
    ensemble_preds = (ensemble_probs > threshold).astype(int)
    return ensemble_preds


def main():
    # ----------------------------
    # 1. Load Training and Test Data
    # ----------------------------
    # Must contain columns: id, text, target
    train_df = pd.read_csv("train.csv")
    # Must contain columns: id, text
    test_df = pd.read_csv("test.csv")

    # ----------------------------
    # 2. Train Individual Models and Get Test Predictions
    # ----------------------------
    print("Training BERT model...")
    bert_test_probs, bert_trainer, bert_tokenizer = train_bert(
        train_df, test_df)

    print("Training CNN model...")
    cnn_test_probs, cnn_model, keras_tokenizer = train_cnn(train_df, test_df)

    # ----------------------------
    # 3. Ensemble Predictions
    # ----------------------------
    ensemble_preds = ensemble_predictions(
        bert_test_probs, cnn_test_probs, threshold=0.5)

    # ----------------------------
    # 4. Save Submission File (maintaining the output format)
    # ----------------------------
    if 'id' in test_df.columns:
        submission = pd.DataFrame(
            {'id': test_df['id'], 'target': ensemble_preds})
    else:
        submission = pd.DataFrame({'target': ensemble_preds})
    submission.to_csv("submission.csv", index=False)
    print("Submission file saved as submission.csv")


if __name__ == "__main__":
    main()
