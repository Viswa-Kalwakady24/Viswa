# model_pipeline.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HF_MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"

# Load model & tokenizer once
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def query_model(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()