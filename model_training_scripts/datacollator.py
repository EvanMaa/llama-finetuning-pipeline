from dataclasses import dataclass
from typing import Dict, List, Union
import torch

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: any
    max_length: int = 2048

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Dynamic padding for inputs
        inputs = self.tokenizer.pad(
            {"input_ids": [x["input_ids"] for x in batch],
             "attention_mask": [x["attention_mask"] for x in batch]},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Pad labels manually with -100
        max_len = inputs["input_ids"].shape[1]
        padded_labels = []
        for x in batch:
            lbl = x["labels"]
            lbl = lbl + [-100] * (max_len - len(lbl))  # pad with -100
            padded_labels.append(lbl[:max_len])

        inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return inputs