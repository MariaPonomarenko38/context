import json, torch, random
from dataclasses import dataclass
from typing import List, Dict
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
# ----------------
# Config
# ----------------
TYPES = ["nationality", "age", "occupation", "education", "location", "public organization", "health", "sexual orientation",
                "finance", "family"]
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ["B","I"]]
BIO2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2BIO = {i: lab for lab, i in BIO2ID.items()}
IMP2ID = {"low": 0, "high": 1}
ID2IMP = {v:k for k,v in IMP2ID.items()}

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)

# ----------------
# Data
# ----------------
class PIIDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def spans_to_bio(question, context, spans):
    """
    Build token-level BIO + importance labels from (question, context, PII dict).
    No explicit start/end indices in data — they are inferred by substring search.
    """
    sep_token = " </s> "
    combined = question.strip() + sep_token + context

    # --- initialize char-level tags ---
    tags = ["O"] * len(combined)

    for pii_val, meta in spans.items():
        pii_val = pii_val.strip()
        pii_type = meta["type"]
        imp = meta["relevance"]

        # find all occurrences of this PII value in combined text
        start = 0
        while True:
            idx = combined.find(pii_val, start)
            if idx == -1:
                break
            end = idx + len(pii_val)
            tags[idx] = f"B-{pii_type}"
            for k in range(idx + 1, end):
                if k < len(tags):
                    tags[k] = f"I-{pii_type}"
            start = end  # continue searching after this occurrence

    # --- tokenize and align to tags ---
    enc = tokenizer(combined, return_offsets_mapping=True, truncation=True, max_length=512)

    labels_bio, labels_imp = [], []
    for (start, end) in enc["offset_mapping"]:
        if start == end:  # special tokens
            labels_bio.append(-100)
            labels_imp.append(-100)
            continue

        # tag lookup
        if start < len(tags):
            lab = tags[start]
            labels_bio.append(BIO2ID.get(lab, BIO2ID["O"]))
        else:
            labels_bio.append(BIO2ID["O"])

        # assign importance only to B- tokens
        if start < len(tags) and tags[start].startswith("B-"):
            pii_type = tags[start].split("-")[1]
            # find corresponding importance
            for v, meta in spans.items():
                if meta["type"] == pii_type:
                    labels_imp.append(IMP2ID[meta["relevance"]])
                    break
            else:
                labels_imp.append(-100)
        else:
            labels_imp.append(-100)

    enc.pop("offset_mapping")
    enc["labels_bio"] = labels_bio
    enc["labels_imp"] = labels_imp
    enc["text"] = combined
    return enc

@dataclass
class Collator:
    def __call__(self, batch: List[Dict]):
        encs = [spans_to_bio(b["question"], b["context"], b["piis"]) for b in batch]
        input_ids      = [torch.tensor(e["input_ids"]) for e in encs]
        attention_mask = [torch.tensor(e["attention_mask"]) for e in encs]
        labels_bio     = [torch.tensor(e["labels_bio"]) for e in encs]
        labels_imp     = [torch.tensor(e["labels_imp"]) for e in encs]

        input_ids      = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels_bio     = pad_sequence(labels_bio, batch_first=True, padding_value=-100)
        labels_imp     = pad_sequence(labels_imp, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_bio": labels_bio,
            "labels_imp": labels_imp,
        }

# ----------------
# Model
# ----------------
class PIIMultiConfig(PretrainedConfig):
    model_type = "pii_multi"
    def __init__(self, base_model="roberta-base", num_bio=len(BIO_LABELS), num_imp=2, **kw):
        super().__init__(**kw)
        self.base_model = base_model
        self.num_bio = num_bio
        self.num_imp = num_imp

class PIIMultiModel(PreTrainedModel):
    config_class = PIIMultiConfig
    def __init__(self, config: PIIMultiConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.base_model)
        hidden = self.encoder.config.hidden_size
        self.bio_head = nn.Linear(hidden, config.num_bio)
        self.imp_head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, config.num_imp))
        self.loss_bio = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_imp = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels_bio=None, labels_imp=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state                     # [B, T, H]
        logits_bio = self.bio_head(H)                 # [B, T, |BIO|]
        logits_imp = self.imp_head(H)                 # [B, T, 2]
        loss = None
        if labels_bio is not None and labels_imp is not None:
            lb = self.loss_bio(logits_bio.view(-1, logits_bio.size(-1)), labels_bio.view(-1))
            li = self.loss_imp(logits_imp.view(-1, logits_imp.size(-1)), labels_imp.view(-1))
            loss = lb + 0.5*li                        # weight importance a bit lower initially
        return {"loss": loss, "logits_bio": logits_bio, "logits_imp": logits_imp}



@torch.no_grad()
def evaluate_f1(model, dataset, collator, device=None, batch_size=8):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds, all_labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch_rows = dataset.rows[i:i+batch_size]
        batch = collator(batch_rows)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out["logits_bio"].argmax(-1).cpu().numpy()
        labels = batch["labels_bio"].cpu().numpy()

        for p, l in zip(preds, labels):
            mask = l != -100
            all_preds.extend(p[mask])
            all_labels.extend(l[mask])

    # exclude 'O' class
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    non_o = all_labels != BIO2ID["O"]

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels[non_o],
        all_preds[non_o],
        average="macro",
        zero_division=0
    )
    return {"precision": prec, "recall": rec, "f1": f1}

@torch.no_grad()
def evaluate_per_type(model, dataset, collator, device=None, batch_size=8):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds, all_labels = [], []
    for i in range(0, len(dataset), batch_size):
        batch_rows = dataset.rows[i:i+batch_size]
        batch = collator(batch_rows)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out["logits_bio"].argmax(-1).cpu().numpy()
        labels = batch["labels_bio"].cpu().numpy()

        for p, l in zip(preds, labels):
            mask = l != -100
            all_preds.extend(p[mask])
            all_labels.extend(l[mask])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    non_o_mask = all_labels != BIO2ID["O"]
    labels_non_o = all_labels[non_o_mask]
    preds_non_o = all_preds[non_o_mask]

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        labels_non_o, preds_non_o,
        labels=list(BIO2ID.values())[1:],  # skip 'O'
        zero_division=0
    )

    result = {}
    for idx, lab_id in enumerate(list(BIO2ID.values())[1:]):
        label_name = ID2BIO[lab_id]
        result[label_name] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx])
        }
    return result
# def evaluate_accuracy(model, dataset, collator, device=None, batch_size=8):
#     """
#     Evaluate BIO and importance token accuracy without using a DataLoader.
#     Uses your custom PIIDataset + Collator.
#     """
#     model.eval()
#     if device is None:
#         device = next(model.parameters()).device

#     total_bio, correct_bio = 0, 0
#     total_imp, correct_imp = 0, 0
#     total_loss = 0

#     for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
#         batch_rows = dataset.rows[i:i+batch_size]
#         batch = collator(batch_rows)
#         batch = {k: v.to(device) for k, v in batch.items()}

#         out = model(**batch)
#         loss = out["loss"]
#         total_loss += loss.item()

#         # BIO predictions
#         preds_bio = out["logits_bio"].argmax(-1)
#         mask = batch["labels_bio"] != -100
#         correct_bio += (preds_bio[mask] == batch["labels_bio"][mask]).sum().item()
#         total_bio += mask.sum().item()

#         # Importance predictions
#         preds_imp = out["logits_imp"].argmax(-1)
#         mask_imp = batch["labels_imp"] != -100
#         correct_imp += (preds_imp[mask_imp] == batch["labels_imp"][mask_imp]).sum().item()
#         total_imp += mask_imp.sum().item()

#     bio_acc = correct_bio / total_bio if total_bio > 0 else 0
#     imp_acc = correct_imp / total_imp if total_imp > 0 else 0
#     avg_loss = total_loss / (len(dataset) / batch_size)

#     return {"loss": avg_loss, "bio_acc": bio_acc, "imp_acc": imp_acc}

def train():
# ----------------
# Train
# ----------------
    train_ds = PIIDataset("./data/train.jsonl")
    #val_ds   = PIIDataset("val.jsonl")

    model = PIIMultiModel(PIIMultiConfig())
    collate = Collator()

    args = TrainingArguments(
        output_dir="pii-detector",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.05,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        remove_unused_columns=False,
    )

    # simple token-level F1 on BIO tags (you can upgrade to span-level later)
    def compute_metrics(eval_pred):
        import numpy as np
        logits_bio, labels_bio = eval_pred.predictions["logits_bio"], eval_pred.label_ids["labels_bio"]
        preds = logits_bio.argmax(-1)
        mask = labels_bio != -100
        tp = (preds[mask] == labels_bio[mask]).sum()
        acc = float(tp) / float(mask.sum().item())
        return {"token_acc": acc}

    class WrapperTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            out = model(**{k: v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels_bio","labels_imp"]})
            loss = out["loss"]
            if prediction_loss_only:
                return (loss, None, None)
            logits = {"logits_bio": out["logits_bio"].detach().cpu().numpy()}
            labels = {"labels_bio": inputs["labels_bio"].detach().cpu().numpy()}
            return (loss, logits, labels)

    trainer = WrapperTrainer(
        model=model, args=args, train_dataset=train_ds, #eval_dataset=val_ds,
        data_collator=collate, tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
    model = PIIMultiModel.from_pretrained("../pii-detector/checkpoint-1995").eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_ds = PIIDataset("./data/train.jsonl")
    test_ds  = PIIDataset("./data/test.jsonl")
    collate  = Collator()

    per_type = evaluate_per_type(model, test_ds, collate, device=device)
    for label, metrics in per_type.items():
        print(label, metrics)

    # move model to GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # train_metrics = evaluate_f1(model, train_ds, collate, batch_size=8, device=device)
    # test_metrics  = evaluate_f1(model, test_ds,  collate, batch_size=8, device=device)

    # print("Train:", train_metrics)
    # print("Test:", test_metrics)

    #train()
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
    # model = PIIMultiModel.from_pretrained("pii-detector/checkpoint-1995").eval()

    # text = "I'm a Guyanese resident frustrated that despite reporting overflowing trash bins last week, CleanCity Solutions still hasn’t addressed the sanitation issue in my neighborhood. This neglect has caused unpleasant odors and attracted pests. As a sanitation supervisor and a devoted parent with two children, I find this situation especially troubling. In my personal life, I recently took up mountain biking and have become an avid bird watcher during my weekend trips. Additionally, I am heterosexual and have been managing chronic sinusitis for years, which makes dealing with strong odors particularly difficult. How can I make sure my community gets timely, effective responses from waste management services when I’ve reported overflowing trash bins but nothing happens and they’re becoming a health hazard?"
    # '''
    # I'm a Guyanese resident frustrated that despite reporting overflowing trash bins last week, CleanCity Solutions still hasn’t addressed the sanitation issue in my neighborhood. This neglect has caused unpleasant odors and attracted pests. As a sanitation supervisor and a devoted parent with two children, I find this situation especially troubling. In my personal life, I recently took up mountain biking and have become an avid bird watcher during my weekend trips. Additionally, I am heterosexual and have been managing chronic sinusitis for years, which makes dealing with strong odors particularly difficult. How can I make sure my community gets timely, effective responses from waste management services when I’ve reported overflowing trash bins but nothing happens and they’re becoming a health hazard? 
    # "piis": {"Guyanese": {"type": "nationality", "relevance": "high", "location": "situation"}, "CleanCity Solutions": {"type": "public organization", "relevance": "high", "location": "situation"}, "two children": {"type": "family", "relevance": "low", "location": "extra facts"}, "sanitation supervisor": {"type": "occupation", "relevance": "low", "location": "extra facts"}, "heterosexual": {"type": "sexual orientation", "relevance": "low", "location": "noise"}, "chronic sinusitis": {"type": "health", "relevance": "low", "location": "noise"}}, 

    # '''

    # # 2) tokenize (KEEP offsets separately)
    # enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
    # offsets = enc["offset_mapping"][0].tolist()   # list of [start,end]
    # inputs  = {k: v for k, v in enc.items() if k != "offset_mapping"}

    # with torch.no_grad():
    #     out = model(**inputs)
    # pred_bio = out["logits_bio"].argmax(-1)[0].tolist()
    # pred_imp = out["logits_imp"].argmax(-1)[0].tolist()
    # print(pred_imp)

    # # 3) decode (pass offsets explicitly)
    # def decode_to_json(text, offsets, pred_bio, pred_imp):
    #     spans = []
    #     i = 0
    #     while i < len(pred_bio):
    #         lab_id = pred_bio[i]
    #         lab = ID2BIO.get(lab_id, "O")

    #         # skip special tokens / padding with empty offsets
    #         if offsets[i][0] == offsets[i][1]:
    #             i += 1
    #             continue

    #         if lab.startswith("B-"):
    #             typ = lab.split("-", 1)[1]
    #             imp = ID2IMP.get(pred_imp[i], "low")
    #             start = offsets[i][0]
    #             end = offsets[i][1]
    #             j = i + 1
    #             while j < len(pred_bio) and ID2BIO.get(pred_bio[j], "O") == f"I-{typ}":
    #                 # also skip any empty-offset tokens inside
    #                 if offsets[j][0] != offsets[j][1]:
    #                     end = offsets[j][1]
    #                 j += 1
    #             value = text[start:end]
    #             spans.append({"value": value, "type": typ, "relevance": imp})
    #             i = j
    #         else:
    #             i += 1

    #     return {s["value"]: {"type": s["type"], "relevance": s["relevance"]} for s in spans}

    # result = decode_to_json(text, offsets, pred_bio, pred_imp)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
  