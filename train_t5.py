import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from math import ceil
import torch
import transformers
print("🤗 Transformers version:", transformers.__version__)

def main():
    # Check GPU availability first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1) Load & split your CSV
    df = pd.read_csv("C:/Users/adity/Projects/Test/gemini_role.csv")
    train_df = df.sample(frac=0.9, random_state=42)
    val_df   = df.drop(train_df.index)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
    })

    # 2) Tokenizer & base model (using Flan-T5)
    MODEL = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Load model directly to GPU if available
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 3) Apply LoRA for parameter-efficient tuning
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_cfg)

    # 4) Preprocess: tokenize inputs & labels
    max_src, max_tgt = 128, 128
    def preprocess(batch):
        src = tokenizer(
            batch["user_prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_src
        )
        tgt = tokenizer(
            batch["role_playing_prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_tgt
        )
        # mask pad tokens in labels
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in tg]
            for tg in tgt["input_ids"]
        ]
        src["labels"] = labels
        return src

    tokenised = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names
    )

     # === DEBUG: inspect one tokenized example ===
    print("Sample tokenized example:")
    print(tokenised["train"][0])
    # Count non-ignored label tokens
    lbls = tokenised["train"][0]["labels"]
    nonpad = sum(1 for t in lbls if t != -100)
    print("Non-ignored label tokens:", nonpad)


    train_ds = tokenised["train"]
    batch_size = 8  # same as per_device_train_batch_size
    steps_per_epoch = ceil(len(train_ds) / batch_size)

    training_args = Seq2SeqTrainingArguments(
        output_dir="t5-roleplay",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        do_train=True,
        do_eval=True,
        num_train_epochs=20,
        eval_strategy="steps",          # evaluate every N steps (one epoch)
        eval_steps=steps_per_epoch,       # use the computed steps_per_epoch
        save_strategy="steps",          # save checkpoint every N steps
        save_steps=steps_per_epoch,       # same as above
        logging_strategy="steps",
        logging_steps=20,

        # Generation settings for Seq2SeqTrainer
        generation_max_length=128,
        generation_num_beams=4,

        # Save best model by eval_loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False
    )


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    
    print("##############################################################")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print("##############################################################")

        # 3.5) DEBUG: check LoRA adapters and dtype
    # ────────────────────────────────────────────────
    # Count how many params require gradients
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔍 Trainable parameters (should be > 0): {num_trainable:,}")

    # Print a couple of parameter names & dtypes to verify they’re on CUDA & fp16
    sample = [n for n, p in model.named_parameters() if p.requires_grad][:5]
    for n in sample:
        p = dict(model.named_parameters())[n]
        print(f" • {n:<40} device={p.device} dtype={p.dtype}")
    # ────────────────────────────────────────────────

    # ==== DEBUG: single‐batch forward pass ====
    batch = next(iter(trainer.get_train_dataloader()))
    # Move batch to the same device as the model
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    print("➡️ Single‐batch loss:", out.loss.item())
    # ===========================================

    
    # 6) Train & save
    trainer.train()
    trainer.save_model("t5-roleplay")
    tokenizer.save_pretrained("t5-roleplay")

if __name__ == "__main__":
    main()