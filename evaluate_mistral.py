import json, re, torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import pipeline

# ========== CONFIG ==========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_PATH  = "./data/test.jsonl"
N_SAMPLES  = 1   # set to small int for debugging, None = full test
MAX_NEW_TOKENS = 350

# ========== LOAD MODEL ==========
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ========== LOAD DATA ==========
with open(TEST_PATH, "r") as f:
    data = [json.loads(line) for line in f]
if N_SAMPLES:
    data = data[:N_SAMPLES]

print(f"Loaded {len(data)} samples.")

# ========== HELPER: SAFE JSON PARSER ==========
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
        prompt = f"""USER: Analyze the text and list all personally identifiable information (PII) with their type and relevance.
        Respond strictly in JSON format where keys are PII spans and values contain {{"type": ..., "relevance": ...}}.

        ### Example 1
        Context:
        I live in Toronto and my sister just got accepted to the University of Waterloo to study computer science.
        Question:
        What advice should I give her about moving to a new city?
        """
        prompt += """ASSISTANT:
        {
        "Toronto": {"type": "location", "relevance": "low"},
        "University of Waterloo": {"type": "education", "relevance": "high"},
        "sister": {"type": "family", "relevance": "low"}
        }"""
        prompt += f"""
        Context:
        {ex['context']}

        Question:
        {ex['question']}

        ASSISTANT:"""

        out = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS,
                   temperature=0.0, do_sample=False)
        text = out[0]["generated_text"]
        text = text[len(prompt):]
        print(text)
        preds.append(try_parse_json(text))
    return preds

# ========== EVALUATION ==========
def evaluate_llm_predictions(preds, gold):
    """
    Evaluate LLM predictions against gold-standard PIIs.

    Args:
        preds: list of dicts, each mapping PII value -> {"type": ..., "relevance": ...}
        gold:  list of dicts, same structure (ground truth)

    Returns:
        dict with F1 metrics for span detection, type classification, and relevance classification
    """
    # -----------------
    # 1️⃣ Span Detection
    # -----------------
    y_true_spans, y_pred_spans = [], []
    for g, p in zip(gold, preds):
        gt_keys = set(g.keys())
        pr_keys = set(p.keys())
        for key in gt_keys:
            y_true_spans.append(1)
            y_pred_spans.append(1 if key in pr_keys else 0)
        for key in pr_keys - gt_keys:
            y_true_spans.append(0)
            y_pred_spans.append(1)
    prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
        y_true_spans, y_pred_spans, average="binary", zero_division=0
    )

    # -----------------
    # 2️⃣ Type Classification (only for correctly detected spans)
    # -----------------
    y_true_type, y_pred_type = [], []
    for g, p in zip(gold, preds):
        for key, gold_info in g.items():
            if key in p:  # only compare if span detected
                y_true_type.append(gold_info["type"].lower())
                y_pred_type.append(p[key].get("type", "").lower())
    labels_type = sorted(set(y_true_type + y_pred_type))
    prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
        y_true_type, y_pred_type, labels=labels_type, average="macro", zero_division=0
    )

    # -----------------
    # 3️⃣ Relevance Classification (only for correctly detected spans)
    # -----------------
    y_true_rel, y_pred_rel = [], []
    for g, p in zip(gold, preds):
        for key, gold_info in g.items():
            if key in p:
                y_true_rel.append(gold_info.get("relevance", gold_info.get("relevance", "low")).lower())
                y_pred_rel.append(p[key].get("relevance", p[key].get("relevance", "low")).lower())
    labels_rel = ["low", "high"]
    prec_r, rec_r, f1_r, _ = precision_recall_fscore_support(
        y_true_rel, y_pred_rel, labels=labels_rel, average="macro", zero_division=0
    )

    # -----------------
    # Combine results
    # -----------------
    return {
        "span_precision": prec_s,
        "span_recall": rec_s,
        "span_f1": f1_s,
        "type_precision": prec_t,
        "type_recall": rec_t,
        "type_f1": f1_t,
        "relevance_precision": prec_r,
        "relevance_recall": rec_r,
        "relevance_f1": f1_r,
    }

# ========== BUILD GOLD LABELS ==========
gold = [{k: {"type": v["type"], "relevance": v["relevance"]}
         for k, v in ex["piis"].items()} for ex in data]

# ========== RUN EVERYTHING ==========
preds = generate_predictions(pipe, data)
metrics = evaluate_llm_predictions(preds, gold)

# ========== REPORT ==========
print("\n=== Evaluation (zero-shot Mistral-7B-Instruct) ===")
print(json.dumps(metrics, indent=2))