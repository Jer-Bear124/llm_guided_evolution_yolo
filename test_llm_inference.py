import sys

import re
import os
import glob
import time
import numpy as np
import transformers
import torch
from torch import bfloat16, float16
from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer

#MODEL="Qwen/Qwen2.5-72B-Instruct"
#MODEL="mixtral/Mixtral-8x7B-Instruct-v0.1"
#MODEL="google/gemma-2-2b-it"
#MODEL="/storage/ice-shared/vip-vvk/llm_storage/Qwen/Qwen2.5-72B-Instruct"
MODEL="/storage/ice-shared/vip-vvk/llm_storage/Qwen/Qwen2.5-7B-Instruct"
#MODEL="/storage/ice-shared/vip-vvk/llm_storage/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#MODEL="/storage/ice-shared/vip-vvk/llm_storage/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

#mistralai

def submit_llm(txt2llm, max_new_tokens=764, top_p=0.15, temperature=0.1, 
               model_id=MODEL, return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)  # Random token limit
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()

    # Load Model from Local Path
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Avoids errors on CPU
        device_map="auto"  # Automatically assigns to GPU if available (Set to cuda)
    )
    model.eval()
    print(f"Model loaded on device: {model.device}")

    # Load Tokenizer from Local Path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # Define Text Generation Pipeline
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # If using LangChain, set to True
        task="text-generation",
        temperature=temperature,  # 'Randomness' of outputs
        top_p=top_p,  # Select from top tokens whose probability sum ≤ 15%
        top_k=0,  # Use `top_p` for token selection
        max_new_tokens=max_new_tokens,  # Max tokens to generate
        repetition_penalty=1.1,  # If output repeats, increase
        do_sample=True,
        #device=model.device  # Ensures the pipeline runs on the correct device
    )

    # Run Text Generation
    print("Starting inference...")
    res = generate_text(txt2llm)
    print("Inference complete.")
    output_txt = res[0]["generated_text"]

    print(f'Time to load and generate: {round(time.time()-start_time)} seconds')

    return (output_txt, generate_text) if return_gen else output_txt

if __name__ == "__main__":
    print("STARTING LLM TEXT GENERATION")
    text2llm = "Write a creative story about a team of college students making a breakthrough in AI."
    print("Input Text: ", text2llm)
    output_txt = submit_llm(text2llm)
    print("LLM OUTPUT: \n", output_txt)