import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# 0) Device setup
device = 0 if torch.cuda.is_available() else -1

# 1) Load the base T5 model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 2) Load your COT adapter from disk
#    Replace this with the actual path to your CoT adapter folder
adapter_path = "C:/Users/adity/Projects/Promptify/t5-cot"

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 3) Load the tokenizer (must point at the adapter folder to get any special tokens)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# 4) Wrap in a pipeline for generation
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,

    # let it stop naturally once it emits </s>
    early_stopping=True,
    do_sample=False,

    # beam settings
    num_beams=4,
    max_length=128,           # your rewrites are ~50-80 tokens
    length_penalty=1.0,       # >=1 discourages overly long outputs
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
)


# 5) Test prompts for chain-of-thought
tasks = [
    "Write a persuasive cover letter for a data analyst position.",
    "Summarize the key findings of a research paper on climate change.",
    "Generate a lesson plan for teaching fractions to 4th graders.",
    "Draft a proposal for implementing renewable energy in a small town.",
    "Create a product description for an innovative smartwatch.",
    "Write a short mystery story set in Victorian London.",
    "Outline a marketing strategy for a new vegan restaurant.",
    "Compose a motivational speech for a high school graduation.",
    "Develop a recipe using only five ingredients.",
    "Design a UX questionnaire for a mobile banking app."
]

for task in tasks:
    output = pipe(task)[0]["generated_text"]
    print(f"> {task}\n→ {output}\n")
