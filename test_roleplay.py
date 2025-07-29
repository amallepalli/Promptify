import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

device = 0 if torch.cuda.is_available() else -1

# 1) Load the base T5 model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 2) Load your adapter from the folder
#    Replace this path with wherever your folder actually lives
adapter_path = "C:/Users/adity/Projects/Promptify/t5-roleplay"

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 3) Load the tokenizer (pulls in any special tokens you added)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 4) Wrap in a pipeline for easy batch inference
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,

    # Allow longer outputs
    max_length=256,
    min_length=80,

    # Use beam search to improve fluency
    num_beams=5,
    length_penalty=0.7,

    # Discourage exact repeats
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,

    # Stop only at EOS or max_length
    early_stopping=False,

    # Keep deterministic decoding (no sampling)
    do_sample=False
)

# 5) Test it
tasks = [
    "Explain blockchain to a 10-year-old",
    "Draft an email asking for project feedback",
    "Explain CRISPR gene editing to a high school biology class",
    "Draft a compelling pitch for a solar-powered scooter startup",
    "Outline a beginner’s workshop on mindfulness meditation",
    "Analyze supply-chain risks for a small organic café",
    "Write a fundraising appeal for a local animal-rescue shelter",
    "Describe the basics of quantum computing to a business-savvy audience",
    "Create a safety briefing on lab-hazard protocols for new research assistants",
    "Develop a mock debate between proponents and critics of universal basic income",
    "Summarize the key differences between agile and waterfall project management",
    "Compose a style-guide prompt for writing friendly customer-service emails"
]

for t in tasks:
    out = pipe(t)[0]["generated_text"]
    print(f"> {t}\n→ {out}\n")
