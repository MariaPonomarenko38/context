#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================================
# CONFIG
# ==========================================================
# Change these paths
BASE_MODEL = "Qwen/Qwen2.5-7B"   # or Qwen2.5, etc.
FINETUNED_DIR = "./models/context-pii-detection-qwen"              # where your trained model or adapter is saved
USE_LORA = True                         # set False if you trained full model

# ==========================================================
# LOAD MODEL & TOKENIZER
# ==========================================================
print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if USE_LORA:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, FINETUNED_DIR)
else:
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

model.eval()
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Model loaded.")

# ==========================================================
# PROMPT / QUESTION
# ==========================================================
question = "You need to find the PIIs in the provided text and classify them by type and relevance.The type can belong to one of these categories: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family The relevance can be either high or low. The relevance score should be decided based on how strongly the PII is related to the question — for example, PIIs directly influencing the question context or needed to answer it are 'high' relevance. Analyze the following text and produce a JSON output with the structure { 'value1': { 'type': ..., 'importance': ...}, 'value2': ...}. Do not give any other explanations.Text: Last month, while volunteering with HealthGuard Solutions, I struggled to address vaccine hesitancy in my community because inconsistent government regulations left many confused and distrustful about immunization programs. As a community health worker with a Bachelor's in Public Health, I am dedicated to improving public health outcomes. Outside of my professional role, I have a keen interest in vintage jazz records and often spend weekends exploring local vinyl shops and attending live music events. Additionally, I have a younger sister who means the world to me. My annual income is $48,000. How can consistent and clear communication help me feel less and build my trust?"

messages = [
    {"role": "user", "content": question},
]

# Build model input using Qwen’s chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# ==========================================================
# GENERATION
# ==========================================================
print("🤖 Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\n==================== MODEL RESPONSE ====================")
print(response)
print("========================================================")