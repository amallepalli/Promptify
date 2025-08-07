import pandas as pd
from transformers import AutoTokenizer

# 1) Load your data
df = pd.read_csv("C:/Users/adity/Projects/Promptify/combined_cot_c&q.csv")

# 2) Initialize the same tokenizer you’ll use for training
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# 3) Compute token lengths
def token_len(text):
    # we don’t truncate here so you get the full length
    return len(tokenizer.encode(text, add_special_tokens=True))

df["token_length"] = df["better_prompt"].apply(token_len)

# 4) Inspect the distribution
print("▶️ Max token length:", df["token_length"].max())
print("▶️ 95th percentile:", df["token_length"].quantile(0.95))
print("▶️ Mean length:", df["token_length"].mean())

# (Optional) See how many will overflow 256:
overflow = (df["token_length"] > 256).sum()
print(f"▶️ Examples over 256 tokens: {overflow} / {len(df)}")
