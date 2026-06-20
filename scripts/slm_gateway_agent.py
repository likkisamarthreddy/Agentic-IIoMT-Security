import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def load_dataset(jsonl_file):
    data = {"prompt": [], "response": []}
    with open(jsonl_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            data["prompt"].append(obj["prompt"])
            data["response"].append(obj["response"])
    return Dataset.from_dict(data)

def main():
    print("Loading synthetic dataset...")
    dataset = load_dataset("slm_synthetic_data.jsonl")
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Loading tokenizer and model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # We use bfloat16 or float16 depending on Kaggle GPU capabilities.
    # bnb_4bit can be configured if bitsandbytes is installed, 
    # but here we follow the standard causal LM workflow.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("Preparing LoRA...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # model = prepare_model_for_kbit_training(model) # Only needed if load_in_4bit=True
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    def tokenize_function(examples):
        texts = [p + " " + r for p, r in zip(examples["prompt"], examples["response"])]
        # Make sure input fits within context length
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./slm_gateway_agent",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True, # enable mixed precision
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
                                    'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
                                    'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])}
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving the LoRA adapter...")
    model.save_pretrained("slm_gateway_agent_lora")
    tokenizer.save_pretrained("slm_gateway_agent_lora")
    print("Done!")

if __name__ == "__main__":
    main()
