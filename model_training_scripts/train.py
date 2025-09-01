import torch
import json
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, LlamaTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from .datacollator import DataCollatorForSupervisedDataset

def tokenizer_fn(example, tokenizer, max_len=2048):
    """
    Tokenize a single example:
      - Builds full string: <s>[INST] <<SYS>>{system}<</SYS>> user [/INST] assistant </s>
      - Masks labels so loss is only on assistant tokens
    """
    # Build the full prompt with system instruction
    system_text = example['system']
    full_text = f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{example['user']} [/INST] {example['assistant']}{tokenizer.eos_token}"
    
    # Tokenize the full text
    enc_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        add_special_tokens=False
    )
    
    # Tokenize the user part (including system) to find where to mask labels
    user_text = f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n\n{example['user']} [/INST]"
    enc_user = tokenizer(user_text, add_special_tokens=False)
    
    # Labels are the same as input_ids, but mask the user part
    labels = enc_full["input_ids"].copy()
    cutoff = len(enc_user["input_ids"])
    labels[:cutoff] = [-100] * cutoff  # Mask user tokens with -100
    
    return {
        "input_ids": enc_full["input_ids"],
        "attention_mask": enc_full["attention_mask"],
        "labels": labels
    }


def tokenize_dataset(jsonl_path, tokenizer_name="meta-llama/Llama-2-7b-chat-hf", max_length=2048):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            tokenized = tokenizer_fn(ex, tokenizer, max_length)
            examples.append(tokenized)

    dataset = Dataset.from_list(examples)
    return dataset

def train_on_sub(sub):

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
    model_name,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
        ),
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
    lora_dropout=0.1,
    bias="none",
    modules_to_save = ["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=f"./models/{sub}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=200,
        fp16=True,
        save_total_limit=3
    )

    train_dataset = tokenize_dataset(f"./data/raw/{sub}_train.jsonl", model_name, 2048)
    val_dataset   = tokenize_dataset(f"./data/raw/{sub}_val.jsonl", model_name, 2048)

    collator = DataCollatorForSupervisedDataset(tokenizer, max_length=2048)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=collator
    )
    
    trainer.train()

def main():
    subname = input("What data set would you like to train on? ")
    train_on_sub(subname)

if __name__ == "__main__":
    main()