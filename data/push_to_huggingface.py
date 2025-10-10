import os
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
import
# ==========================================================
# CONFIGURATION
# ==========================================================
# Change these:
HF_USERNAME = "ponoma16"                # 👈 your Hugging Face username
DATASET_NAME = "context-aware-pii-detection-v3"          # 👈 desired dataset name on the Hub
DATA_DIR = "./"                          # folder with your JSONL files
PRIVATE = False                              # set True if you want private repo

# Paths to splits
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")

# ==========================================================
# CREATE DATASET OBJECTS
# ==========================================================
print("🔹 Loading JSONL splits...")
train_ds = load_dataset("json", data_files={"train": TRAIN_PATH})["train"]
val_ds   = load_dataset("json", data_files={"validation": VAL_PATH})["validation"]
test_ds  = load_dataset("json", data_files={"test": TEST_PATH})["test"]

dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

# ==========================================================
# OPTIONAL: PREVIEW STRUCTURE
# ==========================================================
# ==========================================================
# PUSH TO HUB
# ==========================================================
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

print(f"\n🚀 Pushing dataset to Hugging Face Hub: {repo_id}")
dataset_dict.push_to_hub(repo_id, private=PRIVATE)
print(f"✅ Uploaded successfully! View it at: https://huggingface.co/datasets/{repo_id}")