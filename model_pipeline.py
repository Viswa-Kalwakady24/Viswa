from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HF_MODEL = "ibm-granite/granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def query_model(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text.replace(input_text, "").strip()
    return response