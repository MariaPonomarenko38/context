"""
Example usage:

# With validation loss + early stopping on eval_loss
python -m train_alireza --model_name_or_path Qwen/Qwen2.5-7B \
    --max_length 512 --batch_size 32 --num_epochs 1 \
    --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj --lr 5e-5 \
    --train_file ./data/train.tsv \
    --val_file ./data/val.tsv \
    --skip_header \
    --task_name context-aware-pii-detection \
    --early_stop_patience 2 \
    --early_stop_metric eval_loss



    
# Without validation (no eval loop)
python -m fine_tuning.SFT --model_name_or_path Qwen/Qwen2.5-3B \
    --max_length 512 --batch_size 16 --num_epochs 12 \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj --lr 5e-5 \
    --train_file data/train.tsv \
    --test_file data/test.tsv \
    --skip_header \
    --task_name insurance_policy_interpretation
"""

import argparse
import csv
import wandb
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
#from utils import prompt_utils
import re
import os
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback, 
    TrainerControl,
    TrainerState,
    EarlyStoppingCallback,  # <-- added
)


# ===================== Dataset =====================

class TsvQADataset(Dataset):
    """A Q-A instruction dataset loaded from a TSV file.

    The loader is tolerant to:
        • Arbitrary extra columns → pick `question_col` and `answer_col`.
        • An optional header row → skip when `--skip_header` flag is passed.
    """

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        question_col: int = 0,
        answer_col: int = 2,
        skip_header: bool = False,
        debug_max_items: Optional[int] = None,
        include_reasoning: bool = False,
        reasoning_col: int = 3,
        args=None
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args

        with open(path, newline="", encoding="utf-8") as fp:
            reader = csv.reader(fp, delimiter="\t")

            # Skip header row if requested
            if skip_header:
                next(reader, None)

            for idx, row in enumerate(reader):
                if debug_max_items is not None and idx >= debug_max_items:
                    break
                # Skip empty lines
                if not row or all(cell.strip() == "" for cell in row):
                    continue

                # Robust column access
                if question_col >= len(row) or abs(answer_col) > len(row):
                    raise ValueError(
                        f"Row {idx} has {len(row)} columns but requested q_col={question_col}, a_col={answer_col}."
                    )

                question = row[question_col].strip()
                answer = row[answer_col].strip()

                if include_reasoning:
                    reasoning = row[reasoning_col].strip()
                    answer = reasoning + "\n" + answer
                
                # Prefix prompt
                question = f"{question}"
                prompt = question + "\nAnswer: "

                # Add EOS to target
                answer = answer + tokenizer.eos_token

                # Tokenize (no added specials)
                q_enc = tokenizer(prompt, add_special_tokens=False)
                a_enc = tokenizer(answer, add_special_tokens=False)

                input_ids = q_enc["input_ids"] + a_enc["input_ids"]
                attention_mask = [1] * len(input_ids)

                labels = ([-100] * len(q_enc["input_ids"]) + a_enc["input_ids"])

                # Truncate if needed
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                labels = labels[: self.max_length]

                self.samples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ===================== Collator =====================

class DataCollatorForCausalLMMasking:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        attn_list = [f["attention_mask"] for f in features]

        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding=True,
            return_tensors="pt",
        )
        max_seq_len = batch_input["input_ids"].size(1)

        padded_labels = []
        for lbl in labels_list:
            lbl = lbl + [-100] * (max_seq_len - len(lbl))
            padded_labels.append(lbl)
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)

        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": batch_input["attention_mask"],
            "labels": batch_labels,
        }


# ===================== Evaluation helpers =====================

def _build_answer_patterns(gold_answer: str):
    _TRUE_SYNS = [r"true", r"t", r"yes", r"y", r"correct"]
    _FALSE_SYNS = [r"false", r"f", r"no", r"n", r"incorrect"]

    g = gold_answer.strip()
    g_cf = g.casefold()

    patterns = []
    patterns.append(re.compile(rf"(?i)(?<!\w){re.escape(g_cf)}(?!\w)"))

    if len(g) == 1 and g.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        U = re.escape(g.upper())
        patterns.extend([
            re.compile(rf"(?i)(?:^|\b)[\(\[]?{U}[\)\]\.:]?(?:\b|$)"),
            re.compile(rf"(?i)\b(?:option|choice)\s*{U}\b"),
        ])

    if g_cf in ("true", "yes"):
        patterns.append(re.compile(rf"(?i)(?<!\w)(?:{'|'.join(_TRUE_SYNS)})(?!\w)"))
    elif g_cf in ("false", "no"):
        patterns.append(re.compile(rf"(?i)(?<!\w)(?:{'|'.join(_FALSE_SYNS)})(?!\w)"))

    return patterns

def _matches_gold(pred: str, gold_answer: str) -> bool:
    pred_norm = " ".join(pred.split())
    first_line = pred.splitlines()[0] if pred else ""
    pats = _build_answer_patterns(gold_answer)
    return any(p.search(pred_norm) or p.search(first_line) for p in pats)

# def evaluate_accuracy(
#     model,
#     tokenizer,
#     tsv_path: str,
#     question_col: int = 2,
#     answer_col: int = 1,
#     skip_header: bool = False,
#     max_new_tokens: int = 256,
#     raw_eval: bool = False,
#     task_name: str = "hearsay",
#     last_char_only: bool = False,
#     args=None,
# ):
#     model.eval()

#     total, correct = 0, 0
#     y_true, y_pred = [], []
#     results = []

#     with open(tsv_path, newline="", encoding="utf-8") as fp:
#         reader = csv.reader(fp, delimiter="\t")
#         if skip_header:
#             next(reader, None)

#         for row in tqdm(reader):
#             if not row or all(cell.strip() == "" for cell in row):
#                 continue
#             if question_col >= len(row) or abs(answer_col) > len(row):
#                 continue

#             total += 1
#             question = row[question_col].strip()
#             gold_answer = row[answer_col].strip()

#             if raw_eval:
#                 prompt = (
#                     f"{prompt_utils.SHORT_TASK_DEFINITION_PROMPT_REGISTRY__ZERO_SHOT[task_name]} "
#                     f"Question: {question}\nAnswer: "
#                 )
#                 prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
#             else:
#                 prompt = f"Question: {question}\nAnswer: "

#             enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
#             enc = {k: v.to(model.device) for k, v in enc.items()}

#             gen_ids = model.generate(
#                 **enc,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#             )[0][enc["input_ids"].size(1):]

#             pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#             if last_char_only and pred:
#                 pred = pred[-1]

#             if raw_eval:
#                 is_correct = _matches_gold(pred, gold_answer)
#             else:
#                 is_correct = (pred == gold_answer)

#             if is_correct:
#                 correct += 1

#             y_true.append(gold_answer)
#             y_pred.append(pred)
#             results.append({
#                 "question": question,
#                 "gold": gold_answer,
#                 "pred": pred,
#                 "is_correct": bool(is_correct),
#             })

#     acc = (correct / total) if total else 0.0
#     true_labels = sorted(set(y_true))
#     overall = {"accuracy": acc}

#     if y_true:
#         overall["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
#         aggregates = {}
#         for avg in ("micro", "macro", "weighted"):
#             P, R, F1, _ = precision_recall_fscore_support(
#                 y_true, y_pred, labels=true_labels, average=avg, zero_division=0
#             )
#             aggregates[avg] = {
#                 "precision": float(P),
#                 "recall": float(R),
#                 "f1": float(F1)
#             }
#         overall["aggregates"] = aggregates
#     else:
#         overall.update({
#             "balanced_accuracy": 0.0,
#             "aggregates": {
#                 "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
#                 "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
#                 "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
#             }
#         })

#     per_class = {}
#     if true_labels:
#         p_cls, r_cls, f1_cls, supports = precision_recall_fscore_support(
#             y_true, y_pred, labels=true_labels, average=None, zero_division=0
#         )
#         for cls, p, r, f1, s in zip(true_labels, p_cls, r_cls, f1_cls, supports):
#             per_class[cls] = {
#                 "precision": float(p),
#                 "recall": float(r),
#                 "f1": float(f1),
#                 "support": int(s),
#             }

#     return {
#         "overall": overall,
#         "per_class": per_class,
#         "labels": true_labels,
#         "y_true": y_true,
#         "y_pred": y_pred,
#         "results": results,
#     }


def _sanitize_label_for_wandb(label: str) -> str:
    safe = str(label)
    for ch in ["/", "\\", ":", " ", "\t", "\n"]:
        safe = safe.replace(ch, "_")
    return safe

def _fmt_pct(x):
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return str(x)


# def print_results(epoch, args, metrics):
#     overall = metrics.get("overall", {})
#     per_class = metrics.get("per_class", {})
#     labels = metrics.get("labels", [])
#     y_true = metrics.get("y_true", [])
#     y_pred = metrics.get("y_pred", [])
#     results = metrics.get("results", [])

#     print("✅ Overall metrics:")
#     print(f"   • accuracy          : {_fmt_pct(overall.get('accuracy', 0.0))}")

#     if "balanced_accuracy" in overall:
#         print(f"   • balanced_accuracy : {_fmt_pct(overall['balanced_accuracy'])}")

#     aggregates = overall.get("aggregates")
#     if isinstance(aggregates, dict) and aggregates:
#         print("\n📊 Precision/Recall/F1 by averaging scheme:")
#         header = f"{'average':<10} | {'precision':>9} | {'recall':>6} | {'f1':>6}"
#         bar = "-" * len(header)
#         print(bar)
#         print(header)
#         print(bar)
#         for avg in ("micro", "macro", "weighted"):
#             block = aggregates.get(avg, {})
#             print(
#                 f"{avg:<10} | "
#                 f"{_fmt_pct(block.get('precision', 0.0)):>9} | "
#                 f"{_fmt_pct(block.get('recall', 0.0)):>6} | "
#                 f"{_fmt_pct(block.get('f1', 0.0)):>6}"
#             )
#         print(bar)

#     if per_class:
#         label_width = max(5, min(40, max((len(str(l)) for l in labels), default=5)))
#         header = f"{'class':<{label_width}} | {'precision':>9} | {'recall':>6} | {'f1':>6} | {'support':>7}"
#         bar = "-" * len(header)
#         print("\n🔎 Per-class metrics:")
#         print(bar)
#         print(header)
#         print(bar)
#         for cls in labels:
#             m = per_class.get(cls, {})
#             print(
#                 f"{str(cls):<{label_width}} | "
#                 f"{_fmt_pct(m.get('precision', 0.0)):>9} | "
#                 f"{_fmt_pct(m.get('recall', 0.0)):>6} | "
#                 f"{_fmt_pct(m.get('f1', 0.0)):>6} | "
#                 f"{int(m.get('support', 0)):>7}"
#             )
#         print(bar)

#     os.makedirs(args.output_dir, exist_ok=True)
#     epoch_int = int(epoch) if isinstance(epoch, (int, float)) else 0

#     json_payload = {
#         "epoch": epoch_int,
#         "task_name": getattr(args, "task_name", None),
#         "model_name_or_path": getattr(args, "model_name_or_path", None),
#         "overall": overall,
#         "per_class": per_class,
#         "labels": labels,
#         "results": results
#     }
#     json_path = os.path.join(args.output_dir, f"output_epoch{epoch_int}.json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(json_payload, f, ensure_ascii=False, indent=2)
#     print(f"📝 Saved metrics: {json_path}")

#     if labels and y_true and y_pred:
#         cm = confusion_matrix(y_true, y_pred, labels=labels)

#         plt.figure(figsize=(1.1 + 0.45 * max(3, len(labels)), 1.1 + 0.45 * max(3, len(labels))))
#         plt.imshow(cm, interpolation="nearest")
#         plt.title(f"Confusion Matrix (epoch {epoch_int})")
#         plt.xlabel("Predicted label")
#         plt.ylabel("True label")
#         plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
#         plt.yticks(range(len(labels)), labels)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 plt.text(j, i, str(cm[i, j]), ha="center", va="center")
#         plt.tight_layout()

#         cm_path = os.path.join(args.output_dir, f"confusion_epoch{epoch_int}.jpg")
#         plt.savefig(cm_path, dpi=200, bbox_inches="tight")
#         plt.close()
#         print(f"🖼️  Saved confusion matrix: {cm_path}")

#         try:
#             wandb.log({"confusion_matrix": wandb.Image(cm_path), "epoch": float(epoch)})
#         except Exception:
#             pass

#     try:
#         log_payload = {"epoch": float(epoch)}

#         acc_val = overall.get("accuracy")
#         if isinstance(acc_val, (int, float)):
#             log_payload["eval_accuracy_epoch"] = float(acc_val)
#         for k, v in overall.items():
#             if k == "aggregates":
#                 aggs = v if isinstance(v, dict) else {}
#                 for avg_name, block in aggs.items():
#                     if not isinstance(block, dict):
#                         continue
#                     for metric_name in ("precision", "recall", "f1"):
#                         val = block.get(metric_name)
#                         if isinstance(val, (int, float)):
#                             log_payload[f"eval/aggregates/{avg_name}/{metric_name}"] = float(val)
#                 continue
#             log_payload[f"eval/{k}"] = float(v) if isinstance(v, (int, float)) else v

#         for cls in labels:
#             m = per_class.get(cls, {})
#             cls_key = _sanitize_label_for_wandb(cls)
#             for stat_name in ["precision", "recall", "f1", "support"]:
#                 val = m.get(stat_name)
#                 if val is None:
#                     continue
#                 key = f"eval/per_class/{cls_key}/{stat_name}"
#                 log_payload[key] = float(val) if stat_name != "support" else int(val)

#         try:
#             wandb.save(json_path)
#         except Exception:
#             pass

#         wandb.log(log_payload)
#     except NameError:
#         pass


class CustomEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        current_epoch = int(state.epoch or 0)
        # if current_epoch % 2 == 0 and current_epoch > 4:
        #     print(f"\n📊 Running custom evaluation at epoch {state.epoch:.2f}")

        #     metrics = evaluate_accuracy(
        #         self.model,
        #         self.tokenizer,
        #         tsv_path=self.args.test_file,
        #         question_col=self.args.question_col,
        #         answer_col=self.args.answer_col,
        #         skip_header=self.args.skip_header,
        #         task_name=self.args.task_name,
        #         raw_eval=self.args.raw_eval,
        #         last_char_only=self.args.last_char_only,
        #         max_new_tokens=self.args.max_length,
        #         args=self.args,
        #     )

        #     print_results(state.epoch, self.args, metrics)

        return control


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--train_file", type=str, default="data/legalBench/hearsay/train_split.tsv")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Optional validation TSV file. If set, eval_loss is computed/logged each epoch.")
    parser.add_argument("--test_file", type=str, default="data/legalBench/hearsay/test_split.tsv")
    parser.add_argument("--task_name", type=str, default="hearsay")
    parser.add_argument(
        "--raw_eval",
        action="store_true",
        help="Enable raw evaluation mode when using Instruct models (No fine-tuning). \
            The evaluation will check if the generated text contains the gold answer in a rule-based manner, instead of exact match.",
    )

    # model/tokenizer
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="tsv_sft_ckpts")

    # TSV parsing
    parser.add_argument("--question_col", type=int, default=2, help="Column index for questions (0-based)")
    parser.add_argument("--answer_col", type=int, default=1, help="Column index for answers (-1 means last)")
    parser.add_argument('--reasoning_col', type=int, default=3, help="Column index for reasoning (if applicable)")
    parser.add_argument("--include_reasoning", action="store_true", help="Include reasoning in the answer")
    parser.add_argument("--skip_header", action="store_true", help="Set if the first line is a header row")

    # hyper-params
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--last_char_only', action='store_true')

    # LoRA
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument(
        '--lora_target_modules',
        type=str,
        nargs='+',
        default=['q_proj', 'v_proj'],
        help='Sub-modules to inject LoRA adapters into',
    )

    # misc
    parser.add_argument("--debug_dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # early stopping (optional)
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Enable early stopping if > 0 (number of eval checks to wait). Requires --val_file.")
    parser.add_argument("--early_stop_metric", type=str, default="eval_loss",
                        help="Metric to monitor for early stopping/best model (e.g., eval_loss).")
    parser.add_argument("--early_stop_greater_is_better", action="store_true",
                        help="If set, higher metric is better. Default False (lower is better) for eval_loss.")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # wandb
    wandb.init(project="Qwen-finetuning", name=f"{args.task_name}-{args.model_name_or_path.split('/')[-1]}")
    wandb.config.update(args)

    # logs
    print("Task:", args.task_name)
    print("Model:", args.model_name_or_path)
    print("Train file:", args.train_file)
    if args.val_file:
        print("Val file:", args.val_file)
    print("-------------------------------------------------------")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            target_modules=args.lora_target_modules,
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # datasets
    train_set = TsvQADataset(
        path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
        question_col=args.question_col,
        answer_col=args.answer_col,
        reasoning_col=args.reasoning_col,
        include_reasoning=args.include_reasoning,
        skip_header=args.skip_header,
        debug_max_items=32 if args.debug_dataset else None,
        args=args
    )
    val_set = None
    if args.val_file:
        val_set = TsvQADataset(
            path=args.val_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            question_col=args.question_col,
            answer_col=args.answer_col,
            reasoning_col=args.reasoning_col,
            include_reasoning=args.include_reasoning,
            skip_header=args.skip_header,
            debug_max_items=32 if args.debug_dataset else None,
            args=args
        )

    collator = DataCollatorForCausalLMMasking(tokenizer)

    # training args
    use_eval = val_set is not None
    use_early_stop = use_eval and (args.early_stop_patience > 0)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        logging_steps=5,
        logging_strategy="steps",

        eval_strategy="epoch" if use_eval else "no",
        # save_strategy="epoch" if use_early_stop else "no",
        save_strategy="no",
        save_total_limit=1,                       # <<< keep only 1 checkpoint
        # load_best_model_at_end=use_early_stop,
        load_best_model_at_end=False,
        metric_for_best_model=args.early_stop_metric if use_early_stop else None,
        greater_is_better=args.early_stop_greater_is_better if use_early_stop else None,

        # Disk-light saving (supported on recent Transformers; harmless if ignored)
        save_safetensors=True,
        save_only_model=True,    

        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        report_to=["wandb"],
    )

    # callbacks
    callbacks = [CustomEvalCallback(model, tokenizer, args)]
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience,
            early_stopping_threshold=0.0
        ))

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )
    trainer.train()

    SAVE_DIR = "models/context-pii-detection-qwen"
    trainer.model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ LoRA fine-tuned model saved to {SAVE_DIR}")

    # ==========================================================
    #   PUSH TO HUGGING FACE HUB
    # ==========================================================
    # Make sure you've logged in first:
    # >>> huggingface-cli login
    # repo_name = "ponoma16/Qwen-2b-finetuned"  # change this

    # trainer.push_to_hub(repo_name, commit_message="Initial LoRA fine-tuned model")
    # print(f"🚀 Model pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
    # Final evaluation
    # if args.num_epochs == 0:
    # metrics = evaluate_accuracy(
    #     model,
    #     tokenizer,
    #     tsv_path=args.test_file,
    #     question_col=args.question_col,
    #     answer_col=args.answer_col,
    #     skip_header=args.skip_header,
    #     task_name=args.task_name,
    #     raw_eval=args.raw_eval,
    #     last_char_only=args.last_char_only,
    #     max_new_tokens=args.max_length,
    #     args=args
    # )
    # print_results(0, args, metrics)

    wandb.finish()


if __name__ == "__main__":
    main()