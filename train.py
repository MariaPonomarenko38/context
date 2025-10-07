import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# ==========================================================
#                 BASIC CONFIGURATION
# ==========================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "./data/formatted.jsonl"
OUTPUT_DIR = "./mistral7b-lora-pii"
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LR = 2e-5
EPOCHS = 2

os.environ["WANDB_PROJECT"] = "context-aware-pii-detection"
os.environ["WANDB_MODE"] = "online" 

# ==========================================================
#                 LOAD DATASET
# ==========================================================
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def format_example(ex):
    return {"text": ex["prompt"] + " " + ex["completion"]}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# ==========================================================
#                 LOAD TOKENIZER
# ==========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
#                 LOAD BASE MODEL (4-bit)
# ==========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# ==========================================================
#                 PREPARE FOR LORA TRAINING
# ==========================================================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,                      # rank of LoRA adapters
    lora_alpha=16,             # scaling factor
    target_modules=["q_proj", "v_proj"],  # efficient choice for Mistral
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ==========================================================
#                 TRAINING CONFIGURATION
# ==========================================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=2048,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    dataset_text_field="text",
    bf16=True,
    packing=False,
)

# ==========================================================
#                 TRAINER SETUP
# ==========================================================
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    
)

# ==========================================================
#                 TRAINING
# ==========================================================
trainer.train()

# ==========================================================
#                 SAVE MODEL
# ==========================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA fine-tuned model saved to {OUTPUT_DIR}")