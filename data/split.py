import json
import random

INPUT_PATH = "dataset.jsonl"
TRAIN_PATH = "train.jsonl"
VAL_PATH = "val.jsonl"
TEST_PATH = "test.jsonl"

TEST_RATIO = 0.1       # 10% for test
VAL_RATIO = 0.1        # 10% for validation
SEED = 42

# --- load and shuffle ---
with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} samples.")

random.seed(SEED)
random.shuffle(data)

# --- split ---
n_total = len(data)
n_test = int(n_total * TEST_RATIO)
n_val = int(n_total * VAL_RATIO)
n_train = n_total - n_test - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

# --- save ---
def save_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

save_jsonl(TRAIN_PATH, train_data)
save_jsonl(VAL_PATH, val_data)
save_jsonl(TEST_PATH, test_data)

print(f"✅ Saved {len(train_data)} → {TRAIN_PATH}")
print(f"✅ Saved {len(val_data)} → {VAL_PATH}")
print(f"✅ Saved {len(test_data)} → {TEST_PATH}")