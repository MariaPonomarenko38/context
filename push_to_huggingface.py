#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================================================
# CONFIG
# ==========================================================
HF_USERNAME   = "ponoma16"                     # 👈 your HF username
REPO_NAME     = "context-aware-pii-qwen-v3"    # 👈 model repo name
MODEL_PATH    = "./models/context-pii-detection-qwen"                     # 👈 where your model is saved
IS_LORA       = True                           # 🔁 set to False if it's a full fine-tuned model
BASE_MODEL    = "unsloth/Qwen3-8B-unsloth-bnb-4bit"             # 👈 base model (only needed for LoRA)
PRIVATE       = False                          # True → private repo on the Hub

# Login (run once per environment)
# >>> huggingface-cli login <<<  (in your terminal, not here)

# ==========================================================
# LOAD MODEL & TOKENIZER
# ==========================================================
print("🚀 Loading model from", MODEL_PATH)
if IS_LORA:
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    repo_id = f"{HF_USERNAME}/{REPO_NAME}-lora"
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
    )
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ==========================================================
# PUSH TO HUB
# ==========================================================
print(f"📦 Pushing model to Hugging Face Hub: {repo_id}")
model.push_to_hub(repo_id, private=PRIVATE, commit_message="Upload fine-tuned model 🚀")
tokenizer.push_to_hub(repo_id, private=PRIVATE)

print(f"\n✅ Done! View your model here:")
print(f"https://huggingface.co/{repo_id}")