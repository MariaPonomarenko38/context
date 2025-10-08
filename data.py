import json

def format_example(sample):
    context = sample["context"].strip()
    question = sample["question"].strip()
    piis = sample["piis"]
    new_piis = {}
    for pii in piis.keys():
        new_piis[pii] ={
            "type": piis[pii]["type"],
            "relevance": piis[pii]["relevance"]
        }

    input_text = (
        "USER: Analyze the text and list all personally identifiable information (PII) with their type and relevance based on the question.\n\n"
        "The type should be classified into one of these categories: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.\n"
        "The relevance can be low or high.\n"
        "Output PIIs in the JSON format.\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nASSISTANT:"
    )
    output_text = json.dumps(new_piis, ensure_ascii=False, indent=2)
    return {"prompt": input_text, "completion": output_text}

def format_example_for_pii_detection(sample):
    context = sample["context"].strip()
    question = sample["question"].strip()
    piis = sample["piis"]
    new_piis = ", ".join(list(piis.keys()))

    input_text = (
        "USER: Analyze the text and list all personally identifiable information (PII) spans.\n\n"
        "Output PIIs through comma.\n"
        f"Text:\n{context}\n\nASSISTANT:"
    )
    return {"prompt": input_text, "completion": new_piis}

with open("./data/dataset.jsonl") as f:
    data = [json.loads(line) for line in f]

formatted = [format_example_for_pii_detection(x) for x in data]

with open("./data/formatted_pii_detection.jsonl", "w") as f:
    for ex in formatted:
        f.write(json.dumps(ex) + "\n")