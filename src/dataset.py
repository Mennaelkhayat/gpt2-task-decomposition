"""Dataset for the task decomposition fine-tuning data.

Each example is a high-level task joined with its reference subtasks using
[SEP] separators, exactly as in the original notebook:

    "study english [SEP] learn 20 new words [SEP] practice grammar ..."
"""

import torch
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    def __init__(self, tasks, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        for task in tasks:
            text = task["task"] + " [SEP] " + " [SEP] ".join(task["subtasks"])
            encodings_dict = tokenizer.encode_plus(
                text, truncation=True, max_length=max_length, padding="max_length"
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
