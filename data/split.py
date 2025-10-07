import json, random

# --- config ---
INPUT_PATH = "dataset.jsonl"
TRAIN_PATH = "train.jsonl"
TEST_PATH = "test.jsonl"
TEST_RATIO = 0.1
SEED = 42

# --- load and shuffle ---
with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} samples.")

random.seed(SEED)
random.shuffle(data)

# --- split ---
split_idx = int(len(data) * (1 - TEST_RATIO))
train_data = data[:split_idx]
test_data = data[split_idx:]

# --- save ---
def save_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

save_jsonl(TRAIN_PATH, train_data)
save_jsonl(TEST_PATH, test_data)

print(f"✅ Saved {len(train_data)} → {TRAIN_PATH}")
print(f"✅ Saved {len(test_data)} → {TEST_PATH}")