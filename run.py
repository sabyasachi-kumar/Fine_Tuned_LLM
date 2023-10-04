
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
model_path = "C:/Users/sabya/OneDrive/Desktop/model_final/New folder"  # change to the path where your model is saved
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from transformers import pipeline
prompt = "What is creepage distance?"  # change to your desired prompt
gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = gen(prompt)
print(result[0]['generated_text'])