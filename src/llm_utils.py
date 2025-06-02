import sys
sys.path.append("src")

import re
import os
import glob
import time
import numpy as np
import transformers
import torch #added 2/17/25
from torch import bfloat16
from utils.privit import *
from cfg.constants import *
from utils.print_utils import box_print

from typing import Optional
#import fire
# from llama import Llama
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM #AutoModel added 2/17 to replace InferenceClient


def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()


def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    return '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()


def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, hugging_face=False):
    """Generates augmented code using Mixtral."""
    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)
    print(txt2llm)
    
    if hugging_face is True: #change HUGGING_FACE_BOOL in constants.py
        llm_code_generator = submit_mixtral
        qc_func = llm_code_qc
    else:
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral #not mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        elif LLM_MODEL == 'qwen2.5_7B':
            llm_code_generator = submit_qwen
        qc_func = llm_code_qc_hf
    
    if apply_quality_control:
        base_code = retrieve_base_code(augment_idx)
        code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
        code_from_llm = qc_func(code_from_llm, base_code, generate_text)
    else:
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)
        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        print(code_from_llm)
        code_from_llm = clean_code_from_llm(code_from_llm)
    box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    return code_from_llm

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

# Function to load and split the file
def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, f'{TEMPLATE}/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'{TEMPLATE}/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    #code_from_llm = submit_mixtral_hf(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
    #                  model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    code_from_llm = submit_mixtral_hf(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="/storage/ice-shared/vip-vvk/llm_storage/mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm

def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                         model_path="/storage/ice-shared/vip-vvk/llm_storage/mistralai/Mixtral-8x7B-Instruct-v0.1", 
                         return_gen=False):

    max_new_tokens = np.random.randint(1300, 1500)  # Randomize new tokens (orig. 900, 1300)
    print("Utilizing Mixtral-8x7b-Instruct-v0.1 (submit_mixtral_hf)")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # Load Model & Tokenizer Locally
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Use bfloat16 for GPUs
        device_map="auto"
    ).eval()  # Set to eval mode

    instructions = [
        {"role": "user", "content": "Provide code in Python\n" + txt2mixtral}
    ]
    
    # Convert instructions to model format
    prompt = tokenizer.apply_chat_template(instructions, tokenize=False)

    # Tokenize Input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate Output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode & Return Result
    result_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return (result_text, None) if return_gen else result_text

'''
# Model that uses huggingface API
def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    #huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
'''

def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                        model_path="/storage/ice-shared/vip-vvk/llm_storage/meta-llama/Llama-3.3-70B-Instruct", 
                        return_gen=False):

    max_new_tokens = np.random.randint(1200, 1500)  # Randomize new tokens
    print("Utilizing llama3.3-70B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # Load Model & Tokenizer Locally
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Use bfloat16 for GPUs
        device_map="auto"
    ).eval()  # Set to eval mode

    instructions = [
        {"role": "user", "content": "Provide code in Python\n" + txt2llama}
    ]
    
    # Convert instructions to model format
    prompt = tokenizer.apply_chat_template(instructions, tokenize=False)

    # Tokenize Input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate Output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode & Return Result
    result_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return (result_text, None) if return_gen else result_text

'''
def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="meta-llama/Llama-3.3-70B-Instruct", return_gen=False):
    #"meta-llama/Llama-3.1-70B-Instruct"
    #"meta-llama/Llama-3.3-70B-Instruct"
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    #huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
'''

def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                         model_path="/storage/ice-shared/vip-vvk/llm_storage/mistralai/Mixtral-8x7B-Instruct-v0.1", 
                         return_gen=False):
    max_new_tokens = np.random.randint(1300, 1500)
    print("Utilizing Mixtral-8x7b-Instruct-v0.1 (submit_mixtral)")
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    
    # Use the local model path instead of model_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,  # Forces it to load from the local directory
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    print("LLM OUTPUT")
    print(output_txt)
    print(f'time to load in seconds: {round(time.time()-start_time)}')   

    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text

'''
#This is the function used when pulling from huggingface API
def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
'''

def submit_qwen(txt2smQwen, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                         model_path="/storage/ice-shared/vip-vvk/llm_storage/Qwen/Qwen2.5-7B-Instruct", 
                         return_gen=False):
    max_new_tokens = np.random.randint(2000, 2400)
    print("Utilizing Qwen2.5-7B-Instruct (submit_mixtral)")
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    
    # Use the local model path instead of model_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,  # Forces it to load from the local directory
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        do_sample=True,
    )

    res = generate_text(txt2smQwen)
    output_txt = res[0]["generated_text"]
    print("LLM OUTPUT")
    print(output_txt)
    print(f'time to load in seconds: {round(time.time()-start_time)}')   

    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text




def mutate_prompts(n=5):
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/{TEMPLATE}/FixedPrompts/*/*.txt'), n)
    for i, template in enumerate(templates):
        path, filename = os.path.split(template)
        with open(template, 'r') as file:
            prompt_text = file.read()
        prompt_text = prompt_text.split("```")[0].strip()
        prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_hf
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        output = llm_code_generator(prompt, temperature=temp).strip()
        if "```" in output:
            output = output.split("```")[0]
        output = output + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)