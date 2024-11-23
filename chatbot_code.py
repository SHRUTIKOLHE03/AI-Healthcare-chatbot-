# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:52:51 2024

@author: vishw
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_answer(question, model):
    prompt = "Answer the following question: " + question
    
    tokenizer = AutoTokenizer.from_pretrained("phi1.5")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(input_ids,max_length=512, do_sample=True, top_k=50, top_p=0.9)[0]
    answer = tokenizer.decode(output, skip_special_tokens=True)
    
    end_of_text_index = answer.find("(end of text)")
    if end_of_text_index > -1:
        answer = answer[:end_of_text_index]
    return answer
    

def chatbot(question, history):
    answer = generate_answer(question, llm_model)
    return answer

if __name__ == "__main__":
    # Load the pre-trained model (phi1.5)
    llm_model = AutoModelForCausalLM.from_pretrained("phi1.5", trust_remote_code=True)
    
    interface = gr.ChatInterface(
        chatbot,
        title="AI health assistance",
        description="A general health related question to the AI BOT"
    )
    
    interface.launch()
    
    
#"A general health related question to the AI BOT".find("(end of text)")
