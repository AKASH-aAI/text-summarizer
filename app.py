import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("./save_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./save_summary_model")

def summarize(text):
    text = "summarize: " + text
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=summarize, inputs="text", outputs="text")
iface.launch()
