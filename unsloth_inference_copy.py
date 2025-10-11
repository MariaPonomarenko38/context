import json, torch, re
from tqdm import tqdm
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel

# ========= CONFIG =========
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
TEST_PATH  = "./data/test.jsonl"
SAVE_PATH  = "./data/predicted_pretrained.jsonl"
BATCH_SIZE = 4               # adjust based on VRAM
MAX_NEW_TOKENS = 256         # realistic for JSON generation

# ========= LOAD MODEL =========
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ========= PROMPT TEMPLATE =========
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

instruction_pretrained = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.
Classify their relevance to the question: high, low.

Example:
Text: "John Smith, a 22-year-old student from Canada, works for the University of Toronto."
Question: "What are the educational institutions mentioned in the text?"
Response:
{
  "John Smith": {"type": "family", "relevance": "low"},
  "22-year-old": {"type": "age", "relevance": "low"},
  "student": {"type": "occupation", "relevance": "low"},
  "Canada": {"type": "nationality", "relevance": "low"},
  "University of Toronto": {"type": "education", "relevance": "high"}
}

No explanations or extra text beyond this JSON structure.
"""

# ========= HELPERS =========
def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_block(text: str) -> str:
    """Extracts the first full JSON object {...} from text."""
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
    return ""

# ========= LOAD DATA =========
with open(TEST_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ========= PREBUILD PROMPTS =========
prompts = [
    alpaca_prompt.format(instruction_pretrained, s["context"], s["question"], "")
    for s in data
]

# ========= INFERENCE LOOP (BATCHED) =========
def collate_fn(batch):
    prompts, metas = zip(*batch)
    return list(prompts), list(metas)

loader = DataLoader(
    list(zip(prompts, data)),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,   # ✅ prevent KeyError
)

with open(SAVE_PATH, "w", encoding="utf-8") as fout:
    for batch_prompts, batch_data in tqdm(loader, desc="Generating"):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out, sample, prompt in zip(outputs, batch_data, batch_prompts):
            decoded = tokenizer.decode(out, skip_special_tokens=True)
            decoded = decoded[len(prompt):]
        
            gen_only = extract_json_block(decoded)
            parsed = try_parse_json(gen_only)
      

            record = {
                "context": sample["context"],
                "question": sample["question"],
                "groundtruth": sample.get("piis", {}),
                "decoded": decoded,
                "generated_text": gen_only,
                "parsed": parsed,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
          
print(f"\n✅ Saved all predictions to {SAVE_PATH}")