import json
import csv

INPUT_PATH = "val.jsonl"
OUTPUT_PATH = "val.tsv"

with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

rows = []

for idx, ex in enumerate(data):
    # Combine context and question
    combined_question = f"{ex['context']}"

    # Extract PII dict with type and relevance only
    pii_summary = {
        pii: {
            "type": details["type"],
            "relevance": details["relevance"]
        }
        for pii, details in ex["piis"].items()
    }
    pii_summary = ", ".join([pii for pii, details in ex["piis"].items()])

    instruction = ["You need to find the PIIs in the provided text and output them through comma.",
        "Text: "]
    
    rows.append({
        "index": idx,
        "question": " ".join(instruction) + combined_question,
        "answer": json.dumps(pii_summary, ensure_ascii=False)
    })

# --- write TSV ---
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["index", "question", "answer"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved {len(rows)} rows to {OUTPUT_PATH}")

