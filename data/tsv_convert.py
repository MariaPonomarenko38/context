import json
import csv

INPUT_PATH = "train.jsonl"
OUTPUT_PATH = "train.tsv"

with open(INPUT_PATH, "r") as f:
    data = [json.loads(line) for line in f]

rows = []

for idx, ex in enumerate(data):
    # Combine context and question
    combined_question = f"{ex['context']} {ex['question']}"

    # Extract PII dict with type and relevance only
    pii_summary = {
        pii: {
            "type": details["type"],
            "relevance": details["relevance"]
        }
        for pii, details in ex["piis"].items()
    }

    instruction = ["You need to find the PIIs in the provided text and classify them by type and relevance." 
        "The type can belong to one of these categories: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family",
        "The relevance can be either high or low.",
        "The relevance score should be decided based on how strongly the PII is related to the question —", 
        "for example, PIIs directly influencing the question context or needed to answer it are 'high' relevance.",
        "Analyze the following text and produce a JSON output with the structure { 'value1': { 'type': ..., 'importance': ...}, 'value2': ...}.",
        "Do not give any other explanations."
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

