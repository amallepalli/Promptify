# test_prompt_classifier.py

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ─── Configuration ─────────────────────────────────────────────────────────
MODEL_DIR   = "./prompt-classifier-2"      # path where your classifier was saved
TEST_CSV    = "test_prompts.csv"           # CSV with column "user_prompt"
OUTPUT_CSV  = "test_predictions_3.csv"     # where to write predictions
# ────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load your new/unseen prompts
    df = pd.read_csv(TEST_CSV)

    # 2) Load model and tokenizer
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 3) Bake in human-readable labels
    model.config.id2label = {0: "cot", 1: "roleplay"}
    model.config.label2id = {"cot": 0, "roleplay": 1}

    # 4) Initialize HF text-classification pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False
    )

    # 5) Predict label for each prompt (now returns "cot"/"roleplay")
    df["predicted_label"] = df["user_prompt"].apply(
        lambda x: classifier(x)[0]["label"]
    )

    # 6) Save & print results
    df.to_csv(OUTPUT_CSV, index=False)
    print(df[["user_prompt", "predicted_label"]])

if __name__ == "__main__":
    main()
