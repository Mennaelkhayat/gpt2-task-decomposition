"""Generate subtasks for a new high-level task with the fine-tuned model.

Usage:
    python -m src.generate --task "study english" --model-dir checkpoints/
"""

import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_subtasks(task, model, tokenizer, device, max_length=100):
    model.eval()  # Ensure the model is in evaluation mode
    prompt = f"{task}"
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    # Generation parameters from the original notebook
    sample_outputs = model.generate(
        generated,
        do_sample=True,  # Enable sampling to introduce randomness in the generation
        top_k=30,  # Consider the top 30 tokens at each step for diversity
        max_length=max_length,  # Max length for the generated text
        top_p=0.7,  # Nucleus sampling with p=0.7 for a balance between quality and diversity
        temperature=0.9,  # Slightly less random for more coherent outputs
        num_return_sequences=1,  # Generate one sequence per prompt
        pad_token_id=tokenizer.eos_token_id,  # Use the EOS token as the padding token
    )

    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help='High-level task, e.g. "plan a birthday party"')
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.to(device)

    subtasks = generate_subtasks(args.task, model, tokenizer, device, args.max_length)
    print("Generated Subtasks:", subtasks)


if __name__ == "__main__":
    main()
