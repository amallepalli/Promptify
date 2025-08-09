import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# 0) Device setup
device = 0 if torch.cuda.is_available() else -1

# 1) Load the base T5 + adapter
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base_model, "C:/Users/adity/Projects/Promptify/t5-cot-4.2")
model.eval()

# 2) Load the tokenizer from the adapter folder
tokenizer = AutoTokenizer.from_pretrained("C:/Users/adity/Projects/Promptify/t5-cot-4.2")

# 3) Build your pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,

    # deterministic beams
    do_sample=False,
    early_stopping=True,
    num_beams=4,

    # discourage padding out the output
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,

    # limit *generated* tokens, independent of input length
    max_new_tokens=256,

    # clean up any leftover tokenization artifacts
    clean_up_tokenization_spaces=True,
)

# 4) Run your examples
tasks = [
    "Write a persuasive cover letter for a data analyst position.",
    "How would you summarize the key findings of a research paper on climate change?",
    "Generate a lesson plan for teaching fractions to 4th graders.",
    "What proposal would you draft for implementing renewable energy in a small town?",
    "Create a product description for an innovative smartwatch.",
    "How would you outline a marketing strategy for a new vegan restaurant?",
    "Compose a motivational speech for a high school graduation.",
    "What steps are involved in developing a recipe using only five ingredients?",
    "Draft a UX questionnaire for a mobile banking app.",
    "Explain how to set up a secure home Wi-Fi network."
]

for task in tasks:
    out = pipe(task)[0]["generated_text"].strip()
    print(f"> {task}\n→ {out}\n")
