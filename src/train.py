"""Fine-tune GPT-2 to decompose a high-level task into subtasks.

Usage:
    python -m src.train --epochs 3 --batch-size 4 --output-dir checkpoints/
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_cosine_schedule_with_warmup

from src.dataset import TaskDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/tasks.json")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    tasks = [
        {"task": item["task"], "subtasks": [s["name"] for s in item["subtasks"]]}
        for item in data
    ]

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model)

    dataset = TaskDataset(tasks, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(args.epochs):
        for batch in dataloader:
            input_ids, attn_masks = batch
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model and tokenizer to {args.output_dir}/")


if __name__ == "__main__":
    main()
