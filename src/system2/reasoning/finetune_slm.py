"""
SLM Fine-Tuning Script for IIoMT Agentic Security Framework.

This script implements the PEFT/LoRA fine-tuning for the Phi-3-mini
Small Language Model (SLM) on IIoMT-specific cyber-physical scenarios,
as described in the paper.

It uses the Hugging Face `transformers` and `peft` libraries to apply
Low-Rank Adaptation (LoRA) to the base model, enabling it to accurately
map System 1 alerts to System 2 mitigation playbooks.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

def finetune_slm(
    base_model_id="microsoft/Phi-3-mini-4k-instruct",
    dataset_path="data/iiomt_reasoning_dataset.jsonl",
    output_dir="checkpoints/phi3_iiomt_lora",
    num_train_epochs=3,
    batch_size=4,
    learning_rate=2e-4
):
    print(f"Loading {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Apply LoRA configuration to match the paper's claims
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    except Exception as e:
        print(f"Warning: Failed to load dataset (ensure {dataset_path} exists): {e}")
        print("Skipping training loop due to missing dataset.")
        return
        
    def format_prompt(example):
        return {"text": f"User: {example['instruction']}\nAssistant: {example['output']}"}
        
    dataset = dataset.map(format_prompt)
    
    # SFT Trainer parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        save_strategy="epoch",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    finetune_slm()
