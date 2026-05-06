import argparse
import json
import math
import os
import random
import re
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loader GA unlearning for conformity reduction.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--forget_data_path", type=str, required=True)
    parser.add_argument("--retain_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ga_weight", type=float, default=0.3)
    parser.add_argument("--retain_weight", type=float, default=1.0)
    parser.add_argument("--ga_loss_cap", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every_epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_choices_from_input(text: str):
    pattern = r"^\(([A-Z])\)\s*(.+)$"
    choices = {}
    for line in text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            choices[match.group(1)] = match.group(2).strip()
    return choices


def idx_to_letter(idx: int) -> str:
    return chr(ord("A") + int(idx))


def build_answer_text(choice_idx, prompt: str):
    if choice_idx is None or int(choice_idx) < 0:
        return None
    letter = idx_to_letter(choice_idx)
    choices = extract_choices_from_input(prompt)
    choice_text = choices.get(letter, "").strip()
    if choice_text:
        return f'You: The best answer is: "({letter}) {choice_text}"'
    return f'You: The best answer is: "({letter})"'


class SingleTargetDataset(Dataset):
    def __init__(self, path: str, tokenizer, target_field: str, max_len: int = 2048):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_field = target_field

        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item["input"]
                if target_field == "forget":
                    target = build_answer_text(item.get("guided_pred"), prompt)
                else:
                    target = build_answer_text(item.get("target_label"), prompt)
                if not target:
                    continue
                self.rows.append(
                    {
                        "id": item.get("item_id", ""),
                        "instruction": prompt,
                        "target": target,
                    }
                )
        print(f"Loaded {len(self.rows)} {target_field} samples from {path}")

    def __len__(self):
        return len(self.rows)

    def _format_chat(self, prompt: str, answer: str):
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        full_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False)[0]
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        )[0]
        prompt_len = len(prompt_ids)
        labels = full_ids.clone()
        labels[:prompt_len] = -100
        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]
            labels = labels[:self.max_len]
        attention_mask = torch.ones_like(full_ids)
        return full_ids, labels, attention_mask

    def __getitem__(self, idx):
        item = self.rows[idx]
        input_ids, labels, attention_mask = self._format_chat(item["instruction"], item["target"])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def collate_fn(batch, pad_id: int):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    forget_dataset = SingleTargetDataset(args.forget_data_path, tokenizer, target_field="forget", max_len=args.max_len)
    retain_dataset = SingleTargetDataset(args.retain_data_path, tokenizer, target_field="retain", max_len=args.max_len)
    if len(forget_dataset) == 0 or len(retain_dataset) == 0:
        raise ValueError("Forget dataset or retain dataset is empty.")

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        drop_last=False,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        drop_last=False,
    )

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(forget_loader), len(retain_loader))
    total_optimizer_steps = math.ceil(steps_per_epoch * args.epochs / args.grad_acc)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    loss_log = {"ga": [], "retain": [], "total": [], "step": []}
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        forget_iter = cycle(forget_loader)
        retain_iter = cycle(retain_loader)

        for step_in_epoch in range(steps_per_epoch):
            forget_batch = move_batch_to_device(next(forget_iter), device)
            retain_batch = move_batch_to_device(next(retain_iter), device)

            retain_out = model(**retain_batch)
            loss_retain = retain_out.loss

            forget_out = model(**forget_batch)
            loss_forget = forget_out.loss
            loss_ga = -torch.clamp(loss_forget, max=args.ga_loss_cap)

            total_loss = args.retain_weight * loss_retain + args.ga_weight * loss_ga
            scaled_loss = total_loss / args.grad_acc
            scaled_loss.backward()

            if (step_in_epoch + 1) % args.grad_acc == 0 or step_in_epoch + 1 == steps_per_epoch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    loss_log["ga"].append(float(loss_forget.detach().cpu()))
                    loss_log["retain"].append(float(loss_retain.detach().cpu()))
                    loss_log["total"].append(float(total_loss.detach().cpu()))
                    loss_log["step"].append(global_step)
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"forget_loss={loss_forget.item():.4f} "
                        f"retain_loss={loss_retain.item():.4f} "
                        f"total={total_loss.item():.4f}"
                    )

        if args.save_every_epoch:
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "loss_history.json"), "w", encoding="utf-8") as handle:
        json.dump(loss_log, handle, indent=2)

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"Saved GA LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
