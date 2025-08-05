import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import gradio as gr

checkpoint = "cortecs/Meta-Llama-3-8B-Instruct-GPTQ-8b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

# Load GPTQ model 

model = AutoGPTQForCausalLM.from_quantized(
    checkpoint,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
)

# Function to format prompt + generate response
def predict(message, history):
    prompt = f"<s>[INST] {message.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("[/INST]")[-1].strip()
    return response

# Launch Gradio chatbot
gr.ChatInterface(predict, title=" LLaMA 3 Chatbot").launch(debug=True)
