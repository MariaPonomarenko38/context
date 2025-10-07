import json

def format_example(sample):
    context = sample["context"].strip()
    question = sample["question"].strip()
    piis = sample["piis"]

    input_text = (
        "USER: Analyze the text and list all personally identifiable information (PII) with their type and importance.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nASSISTANT:"
    )
    output_text = json.dumps(piis, ensure_ascii=False, indent=2)
    return {"prompt": input_text, "completion": output_text}

with open("./data/dataset.jsonl") as f:
    data = [json.loads(line) for line in f]

formatted = [format_example(x) for x in data]

with open("./data/formatted.jsonl", "w") as f:
    for ex in formatted:
        f.write(json.dumps(ex) + "\n")