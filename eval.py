
import argparse
import json
import os
import re
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_RE = re.compile(r"\(([A-Z])\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a merged model on a fixed test manifest.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_manifest_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_items", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def idx_to_letter(idx: int) -> str:
    return chr(ord("A") + int(idx))


def letter_to_idx(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def replace_prompt_question(base_prompt: str, injected_question: Optional[str]) -> str:
    if not injected_question:
        return base_prompt
    if "Q:" not in base_prompt:
        return base_prompt
    prefix, suffix = base_prompt.split("Q:", 1)
    suffix_parts = suffix.split("\n\nAnswer choices:", 1)
    if len(suffix_parts) != 2:
        return base_prompt
    return prefix + "Q: " + injected_question.strip() + "\n\nAnswer choices:" + suffix_parts[1]


def build_raw_prompt(item: Dict) -> str:
    return item["input"]


def build_trust_prompt(item: Dict, round_id: int) -> str:
    base_prompt = item["input"]
    guided_outputs = item.get("guided_outputs", {}).get("trust", {})
    guided_answer = guided_outputs.get(str(round_id))
    if guided_answer is None:
        guided_answer = guided_outputs.get(round_id)

    if guided_answer is None:
        return base_prompt

    injected = (
        "You are invited to participate in a Q&A. Here are the current Q&A content:\n\n"
        f"{base_prompt.split('Q:', 1)[1].strip()}\n\n"
        f"Another participant answered before you: {guided_answer}\n\n"
        "Please give your answer in the following format without any other information:\n"
        'You: The best answer is: "(X) the content of the answer"'
    )
    return injected


def extract_pred_letter(text: str) -> Optional[str]:
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def generate_answer(model, tokenizer, prompt: str, args: argparse.Namespace, device: str) -> Dict:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    pred_letter = extract_pred_letter(text)
    pred_idx = letter_to_idx(pred_letter) if pred_letter is not None else -1
    return {
        "model_output": text,
        "pred_letter": pred_letter,
        "pred_idx": pred_idx,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    device = next(model.parameters()).device
    model.eval()

    rows = list(load_jsonl(args.test_manifest_path))
    if args.max_items is not None:
        rows = rows[: args.max_items]

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows, start=1):
            raw_prompt = build_raw_prompt(item)
            raw_result = generate_answer(model, tokenizer, raw_prompt, args, device)

            trust_results = {}
            trust_rounds = item.get("guided_outputs", {}).get("trust", {})
            available_rounds = sorted(int(k) for k in trust_rounds.keys())
            for round_id in available_rounds:
                trust_prompt = build_trust_prompt(item, round_id)
                trust_results[str(round_id)] = generate_answer(model, tokenizer, trust_prompt, args, device)

            output_row = {
                "item_id": item["item_id"],
                "task_name": item["task_name"],
                "item_index": item["item_index"],
                "target_label": item["target_label"],
                "raw_is_correct_reference": item.get("raw_is_correct"),
                "is_forget_item": item.get("is_forget_item"),
                "is_retain_item": item.get("is_retain_item"),
                "raw": {
                    "prompt": raw_prompt,
                    **raw_result,
                },
                "trust": {
                    round_id: {
                        "prompt": build_trust_prompt(item, int(round_id)),
                        **result,
                    }
                    for round_id, result in trust_results.items()
                },
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

            if idx % 20 == 0:
                print(f"Processed {idx}/{len(rows)} items")

    print(f"Saved predictions to {args.output_path}")


if __name__ == "__main__":
    main()