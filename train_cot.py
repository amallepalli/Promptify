import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from math import ceil
import torch
import transformers

print("🤗 Transformers version:", transformers.__version__)

def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Load & split your CoT CSV
    #    CSV columns: user_prompt, answer_cot
    df = pd.read_csv("C:/Users/adity/Projects/Promptify/1200_w_addition.csv")
    train_df = df.sample(frac=0.9, random_state=42)
    val_df   = df.drop(train_df.index)
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
    })

    # 3) Tokenizer & base model
    MODEL = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # 4) LoRA on encoder + decoder
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o", "wi", "wo"]
    )
    model = get_peft_model(model, peft_cfg)

    # 5) Preprocess: raw prompt → optimized (CoT-style) prompt
    max_src, max_tgt = 128, 256
    def preprocess(batch):
        # tokenize the bare question
        src = tokenizer(
            batch["user_prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_src
        )
        # tokenize your target “better_prompt” (the optimized prompt)
        tgt = tokenizer(
            batch["better_prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_tgt
        )
        # mask pad tokens for loss
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in tgt["input_ids"]
        ]
        src["labels"] = labels
        return src

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names
    )

    # 6) Training arguments
    batch_size = 8
    steps_per_epoch = ceil(len(tokenized["train"]) / batch_size)
    training_args = Seq2SeqTrainingArguments(
        output_dir="t5-cot",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        num_train_epochs=15,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        logging_strategy="steps",
        logging_steps=20,

        predict_with_generate=True,
        generation_max_length=max_tgt,
        generation_num_beams=4,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) Sanity-check trainable params
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔍 Trainable parameters: {num_trainable:,}")

    # 8) Train & save
    trainer.train()
    trainer.save_model("t5-cot-4.2")
    tokenizer.save_pretrained("t5-cot-4.2")

if __name__ == "__main__":
    main()
