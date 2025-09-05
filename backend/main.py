import os
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    pipeline,
)
from peft import PeftModel

# ---------------------------
# Config (env or defaults)
# ---------------------------
BASE_T5_NAME = os.getenv("BASE_T5_NAME", "google/flan-t5-base")
ADAPTER_COT_DIR = os.getenv("ADAPTER_COT_DIR", "./adapters/cot")
ADAPTER_ROLEPLAY_DIR = os.getenv("ADAPTER_ROLEPLAY_DIR", "./adapters/roleplay")
CLASSIFIER_DIR = os.getenv("CLASSIFIER_DIR", "./classifier")

DEVICE = 0 if torch.cuda.is_available() else -1

# ---------------------------
# App + CORS
# ---------------------------
app = FastAPI(title="Promptify API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load base models (separate bases to avoid adapter clashes)
# ---------------------------
print("[Boot] Loading base T5 models...")
t5_base_for_cot = AutoModelForSeq2SeqLM.from_pretrained(BASE_T5_NAME)
t5_base_for_roleplay = AutoModelForSeq2SeqLM.from_pretrained(BASE_T5_NAME)

# ---------------------------
# Attach adapters + load tokenizers FROM ADAPTER DIRS (exactly as in tests)
# ---------------------------
print("[Boot] Attaching CoT adapter + tokenizer...")
t5_cot = PeftModel.from_pretrained(t5_base_for_cot, ADAPTER_COT_DIR)
t5_cot.eval()
tok_cot = AutoTokenizer.from_pretrained(ADAPTER_COT_DIR)

print("[Boot] Attaching Roleplay adapter + tokenizer...")
t5_roleplay = PeftModel.from_pretrained(t5_base_for_roleplay, ADAPTER_ROLEPLAY_DIR)
t5_roleplay.eval()
tok_roleplay = AutoTokenizer.from_pretrained(ADAPTER_ROLEPLAY_DIR)

# ---------------------------
# Build TEXT2TEXT pipelines with EXACT generation params from your test scripts
# ---------------------------
print("[Boot] Building CoT pipeline (exact tested settings)...")
pipe_cot = pipeline(
    "text2text-generation",
    model=t5_cot,
    tokenizer=tok_cot,
    device=DEVICE,
    # CoT exact settings (deterministic beams)
    do_sample=False,
    early_stopping=True,
    num_beams=4,
    length_penalty=1.2,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    max_new_tokens=256,
    clean_up_tokenization_spaces=True,
)

print("[Boot] Building Roleplay pipeline (exact tested settings)...")
pipe_roleplay = pipeline(
    "text2text-generation",
    model=t5_roleplay,
    tokenizer=tok_roleplay,
    device=DEVICE,
    # Roleplay exact settings (deterministic beams)
    do_sample=False,
    early_stopping=False,
    num_beams=5,
    length_penalty=0.7,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    min_length=80,
    max_length=256,
)

# ---------------------------
# Classifier
# ---------------------------
print("[Boot] Loading classifier...")
clf_tok = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
clf_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR)
clf: TextClassificationPipeline = pipeline(
    "text-classification",
    model=clf_model,
    tokenizer=clf_tok,
    device=DEVICE,
)

Label = Literal["cot", "roleplay"]

def classify_prompt(prompt: str) -> Label:
    res = clf(prompt)[0]
    label = res["label"].lower()
    if label not in {"cot", "roleplay"}:
        # Adjust mapping if your classifier exported LABEL_0/LABEL_1
        label = "cot" if "0" in res["label"] else "roleplay"
    return label  # type: ignore

# ---------------------------
# Schemas
# ---------------------------
class ClassifyIn(BaseModel):
    prompt: str

class ClassifyOut(BaseModel):
    label: Label

class GenerateIn(BaseModel):
    prompt: str
    adapter_override: Optional[Label] = None  # force a mode if desired

class GenerateOut(BaseModel):
    adapter: Label
    output: str

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyOut)
def classify_route(body: ClassifyIn):
    return {"label": classify_prompt(body.prompt)}

@app.post("/generate", response_model=GenerateOut)
def generate_route(body: GenerateIn):
    mode: Label = body.adapter_override or classify_prompt(body.prompt)

    if mode == "cot":
        text = pipe_cot(body.prompt)[0]["generated_text"].strip()
    else:
        text = pipe_roleplay(body.prompt)[0]["generated_text"].strip()

    return {"adapter": mode, "output": text}
