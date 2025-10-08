import json, re, torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from outlines import from_transformers, Generator

# ========== CONFIG ==========
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_MODEL = "mistral7b-lora-pii"   # path or HF repo for your LoRA adapter
TEST_PATH  = "./data/test.jsonl"
N_SAMPLES  = 1                      # None = full test
MAX_NEW_TOKENS = 350

# ========== LOAD TOKENIZER ==========
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ========== LOAD BASE + LORA ==========
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()

# Merge LoRA weights into the base model (optional but speeds up inference)
model = model.merge_and_unload()

# Build pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ========== LOAD DATA ==========
with open(TEST_PATH, "r") as f:
    data = [json.loads(line) for line in f]
if N_SAMPLES:
    data = data[:N_SAMPLES]
print(f"Loaded {len(data)} samples.")

# ========== SAFE JSON PARSER ==========
def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    s = match.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        try:
            return json.loads(s)
        except Exception:
            return {}

# ========== GENERATION ==========
def generate_predictions(pipe, samples):
    preds = []
    for ex in tqdm(samples, desc="Generating"):
        # prompt = (
        #     "USER: Analyze the text and list all personally identifiable information (PII) with their type and relevance based on the question.\n\n"
        #     "The type should be one of: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.\n"
        #     "The relevance can be 'low' or 'high'.\n"
        #     "Respond strictly in JSON format.\n\n"
        #     f"Context:\n{ex['context']}\n\nQuestion:\n{ex['question']}\n\nASSISTANT:"
        # )
        prompt = (
            "USER: Analyze the text and list all personally identifiable information (PII) spans.\n\n"
            "Output PIIs through comma.\n"
            f"Text:\n{ex['context']}\n\nASSISTANT:"
        )
        print(prompt)
        out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS,
                   temperature=0.0, do_sample=False)
        text = out[0]["generated_text"]
        preds.append(try_parse_json(text))
    return preds

# ========== EVALUATION ==========
def evaluate_llm_predictions(preds, gold):
    y_true, y_pred = [], []
    for g, p in zip(gold, preds):
        gt_keys = set(g.keys())
        pr_keys = set(p.keys())
        # positives (ground truth)
        for key in gt_keys:
            y_true.append(1)
            y_pred.append(1 if key in pr_keys else 0)
        # false positives
        for key in pr_keys - gt_keys:
            y_true.append(0)
            y_pred.append(1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {"precision": prec, "recall": rec, "f1": f1}

# ========== BUILD GOLD LABELS ==========
gold = [{k: {"type": v["type"], "relevance": v["relevance"]}
         for k, v in ex["piis"].items()} for ex in data]

# ========== RUN EVERYTHING ==========
preds = generate_predictions(pipe, data)
metrics = evaluate_llm_predictions(preds, gold)

# ========== REPORT ==========
print("\n=== Evaluation (zero-shot Mistral-7B")
