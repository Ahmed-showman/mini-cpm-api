import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from PIL import Image

base_model_name = "openbmb/MiniCPM-V-2"
lora_path = "./OCRMOD"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

def predict(image):
    image = image.convert("RGB")

    msgs = [{
        "role": "user",
        "content": "Extract ONLY the drug name from this image."
    }]

    with torch.no_grad():
        res = model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=tokenizer
        )

    return res

gr.Interface(fn=predict, inputs="image", outputs="text").launch()