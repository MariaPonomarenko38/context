import json
from tqdm import tqdm

# ========= CONFIG =========
PRED_PATH = "./data/predicted_pretrained.jsonl"

# ========= HELPERS =========
def flatten_gold(piis_dict):
    """Convert dict of dicts -> list of tuples (span, type, relevance)."""
    items = []
    for span, vals in piis_dict.items():
        items.append((span.strip().lower(), vals["type"].lower(), vals["relevance"].lower()))
    return items

def flatten_pred(pred_json):
    """Flatten model output JSON into same (span, type, relevance) tuples."""
    items = []
    try:
        for p in pred_json.keys():
            span = str(p).strip().lower()
            typ = str(pred_json[p].get("type", "")).strip().lower()
            rel = str(pred_json[p].get("relevance", "")).strip().lower()
            if span:
                items.append((span, typ, rel))
    except:
        return items
    return items


def safe_div(a, b):
    return a / b if b > 0 else 0

def f1(p, r):
    return safe_div(2*p*r, p+r) if (p+r) else 0

def compute_scores(pred_tuples, gold_tuples):
    """Compute span, type, relevance precision/recall/F1."""
    # Correct matches
    correct_spans = sum(pred[0] in [g[0] for g in gold_tuples] for pred in pred_tuples)
    correct_types = sum(pred[1] == gold[1] for pred in pred_tuples for gold in gold_tuples if pred[0] == gold[0])
    correct_rels  = sum(pred[2] == gold[2] for pred in pred_tuples for gold in gold_tuples if pred[0] == gold[0])

    # Counts
    n_pred, n_gold = len(pred_tuples), len(gold_tuples)

    span_prec = safe_div(correct_spans, n_pred)
    span_rec  = safe_div(correct_spans, n_gold)
    type_prec = safe_div(correct_types, n_pred)
    type_rec  = safe_div(correct_types, n_gold)
    rel_prec  = safe_div(correct_rels, n_pred)
    rel_rec   = safe_div(correct_rels, n_gold)

    return {
        "span_precision": span_prec,
        "span_recall": span_rec,
        "span_f1": f1(span_prec, span_rec),
        "type_precision": type_prec,
        "type_recall": type_rec,
        "type_f1": f1(type_prec, type_rec),
        "relevance_precision": rel_prec,
        "relevance_recall": rel_rec,
        "relevance_f1": f1(rel_prec, rel_rec),
    }

# ========= LOAD PREDICTIONS =========
results = []
with open(PRED_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Evaluating"):
        sample = json.loads(line)
        gold_tuples = flatten_gold(sample.get("groundtruth", {}))
        pred_tuples = flatten_pred(sample.get("parsed", {}))
        print(gold_tuples)
        print(pred_tuples)
        results.append(compute_scores(pred_tuples, gold_tuples))
        break

# ========= AGGREGATE =========
def avg(key):
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals)/len(vals) if vals else 0

final = {
    "span_precision": avg("span_precision"),
    "span_recall": avg("span_recall"),
    "span_f1": avg("span_f1"),
    "type_precision": avg("type_precision"),
    "type_recall": avg("type_recall"),
    "type_f1": avg("type_f1"),
    "relevance_precision": avg("relevance_precision"),
    "relevance_recall": avg("relevance_recall"),
    "relevance_f1": avg("relevance_f1"),
}

print("\n==== FINAL METRICS (Fine-tuned model) ====")
for k,v in final.items():
    print(f"{k:25s}: {v:.4f}")