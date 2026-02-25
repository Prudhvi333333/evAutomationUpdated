import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "bigscience/bloom-1b1"
lora_model = "DarkMidoriya49/bloom-1b1-lora-tagger"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model)

model.save_pretrained("./bloom-1b1-merged")
tokenizer.save_pretrained("./bloom-1b1-merged")
