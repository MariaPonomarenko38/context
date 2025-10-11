from unsloth import FastLanguageModel
import json
from tqdm import tqdm
import torch
import re
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

instruction = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.
Classify their relevance to the question: high, low.
Output result in the JSON format.
"""

instruction_pretrained = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.
Classify their relevance to the question: high, low.

Example:
Text: "John Smith, a 22-year-old student from Canada, works for the University of Toronto."
Question: "What are the educational institutions mentioned in the text?"
Output:
{
  "John Smith": {"type": "family", "relevance": "low"},
  "22-year-old": {"type": "age", "relevance": "low"},
  "student": {"type": "occupation", "relevance": "low"},
  "Canada": {"type": "nationality", "relevance": "low"},
  "University of Toronto": {"type": "education", "relevance": "high"}
}

No explanations or extra text beyond this JSON structure.
"""

# ========= JSON PARSER =========
def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_block(text: str) -> str:
    """
    Extracts the first full JSON object {...} from the text.
    Works even if there is extra text before/after.
    Returns the JSON substring, or an empty string if not found.
    """
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1].strip()

    # If we never closed all braces
    return ""

def extract_valid_pii_objects(text):
    """
    Extracts all valid {"value": ..., "type": ..., "relevance": ...} objects
    from potentially malformed model outputs.
    Returns a list of dicts.
    """
    # Match independent JSON-like dicts
    pattern = r'\{[^{}]*?"(?:value|span)"\s*:\s*".*?"[^{}]*?"type"\s*:\s*".*?"[^{}]*?"relevance"\s*:\s*".*?"[^{}]*?\}'
    matches = re.findall(pattern, text, re.DOTALL)

    objs = []
    for m in matches:
        try:
            # Clean trailing commas or stray tokens
            m_clean = re.sub(r',\s*([\]}])', r'\1', m)
            obj = json.loads(m_clean)
            objs.append(obj)
        except json.JSONDecodeError:
            continue
    return objs

# ========= LOAD TEST DATA =========
with open('./data/test.jsonl', "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ========= GENERATE =========
with open('./data/predicted_pretrained.jsonl', "w", encoding="utf-8") as fout:
    for sample in tqdm(data):
        # Build input
        input_str = alpaca_prompt.format(
            instruction_pretrained,
            sample["context"],
            sample["question"],
            "",
        )

        # Tokenize and move to GPU
        inputs = tokenizer([input_str], return_tensors="pt").to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048)

        # Decode full output
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt prefix
        gen_only = extract_json_block(decoded)#decoded[len(input_str):].strip()
        
        print(gen_only)

        # Try parsing JSON
        parsed = try_parse_json(gen_only)
        print(parsed)
        print('+++++++')

        # Save result
        record = {
            "context": sample["context"],
            "question": sample["question"],
            "groundtruth": sample.get("piis", {}),
            "decoded": decoded,
            "generated_text": gen_only,
            "parsed": parsed,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✅ Saved all predictions")