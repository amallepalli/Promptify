# train_prompt_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_CSV    = "classifier_training_data.csv"           # ← set path to your CSV
MODEL_NAME  = "distilbert-base-uncased"       # ← or any HF model ID
OUTPUT_DIR  = "./prompt-classifier-2"           # ← where to save model
EPOCHS      = 8
BATCH_SIZE  = 16
SEED        = 42
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load and encode
    df = pd.read_csv(DATA_CSV)
    dataset = Dataset.from_pandas(
    df.rename(columns={"technique": "label"}).drop(columns=["id"]),
    preserve_index=False
    )
    dataset = dataset.class_encode_column("label")  # maps labels to IDs

    # 2) Stratified train/val/test split (80/10/10)
    split1 = dataset.train_test_split(
        test_size=0.10, stratify_by_column="label", seed=SEED
    )
    split2 = split1["test"].train_test_split(
        test_size=0.50, stratify_by_column="label", seed=SEED
    )
    train_ds = split1["train"]
    val_ds   = split2["train"]
    test_ds  = split2["test"]

    # 3) Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_fn(examples):
        return tokenizer(examples["user_prompt"], truncation=True)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # 4) Model init
    num_labels = dataset.features["label"].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )

    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",      
        save_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=BATCH_SIZE,   
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,               
        learning_rate=3e-5,               # lower LR
        weight_decay=0.01,                # small L2 regularization
        warmup_ratio=0.1,                 # 10% warmup
        eval_steps=100,                   # evaluate every 100 steps
        save_steps=100,                   # save every 100 steps
        logging_steps=50,                 # log every 50 steps
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )


    # 6) Metrics
    accuracy = load_metric("accuracy")
    f1       = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 8) Train & evaluate
    trainer.train()
    print("Test set results:", trainer.evaluate(test_ds))

    # 9) Save artifacts
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
