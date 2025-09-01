import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SPECIAL_TOKENS = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]

def build_prompt(user_input, system_prompt=""):
    # The system prompt sets the context and rules for the assistant.
    if system_prompt:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
    else:
        return f"<s>[INST] {user_input} [/INST]"

def chat_with_sub(sub: str):
    """
    Interactive chat session with a PEFT adapter fine-tuned on a subreddit.
    """

    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    base_path = f"./models/{sub}"

    # Find latest checkpoint
    checkpoints = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and re.match(r"checkpoint-\d+", d)
    ]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        adapter_checkpoint = os.path.join(base_path, latest_checkpoint)
        print("Using latest checkpoint:", adapter_checkpoint)
    else:
        raise ValueError("No checkpoints found in the directory.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + 4-bit PEFT adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": 0} if device.type == "cuda" else None
    print("Using device:", device)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )

    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    model.eval()
    model.to(device)

    print("\nChat session started! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        full_prompt = build_prompt(sub, user_input) + tokenizer.eos_token

        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Match training max length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Clamp token IDs to avoid CUDA indexing errors
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=model.config.vocab_size-1)
        inputs["attention_mask"] = inputs["attention_mask"]

        if inputs["input_ids"].shape[1] < 2:
            print("Input too short, please provide more text.\n")
            continue

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=1.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant reply only
        split_markers = ["[/INST]", "User:"]
        for marker in split_markers:
            if marker in response:
                response = response.split(marker)[-1].strip()

        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    sub = input("Which subreddit model do you want to prompt? ")
    chat_with_sub(sub)