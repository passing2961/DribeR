import os
import sys
import copy
import json
import time
import requests

import openai
import together
import torch
import ray
from tqdm import tqdm
from rich.console import Console
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AutoModelForCausalLM
)
import concurrent.futures

from config import ex
from utils import fixed_seed
from load_model import get_conversation_template
from load_dataset import prepare_prompt_dataset

console = Console()
error_console = Console(stderr=True, style='bold red')


mode_to_api_caller = {
    'chat': openai.ChatCompletion,
    'completion': openai.Completion,
}


def call_gpt3(inputs, mode, model_name, params):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    
    openai.organization = os.getenv("OPENAI_ORGANIZATION_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    while not received:
        try:
            if mode == 'chat':
                response = mode_to_api_caller[mode].create(model=model_name, messages=inputs, **params)
            elif mode == 'completion':
                response = mode_to_api_caller[mode].create(model=model_name, prompt=inputs, **params)
            received = True
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g., prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{inputs}\n\n")
                assert False
            
            time.sleep(2)
    
    return response



def generate_together(prompt, config):
    together.api_key = os.getenv("TOGETHER_API_KEY")
    system_prompt = (
        "You are a helpful assistant."
    )
    if 'llama-2' in config['model']['id']:
        prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n" + prompt + "[/INST]"
        prompt = f"User: {prompt} Assistant:"
    elif 'vicuna' in config['model']['id']:
        prompt = f"<s> {system_prompt}\n\n" + prompt
        prompt = f"User: {prompt} Assistant:"
    elif 'mistral' in config['model']['id']:
        prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n" + prompt + "[/INST]"
        

    while True:
        try:
            output = together.Complete.create(
                prompt = prompt, 
                model = config['model']['name'],  
                max_tokens = config['together_params']['max_tokens'],
                temperature = config['together_params']['temperature'],
                top_k = config['together_params']['top_k'],
                top_p = config['together_params']['top_p'],
                repetition_penalty = config['together_params']['repetition_penalty'],
                stop = ['</s>']
            )
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.JSONDecodeError) as e:
            print("Error: {}\nRetrying...".format(e))
            time.sleep(2)
            continue

    return output

def run_together(dataset, config):
    outputs = []
    for instance in tqdm(dataset, total=len(dataset)):
        prompt = instance['prompt']
        while True:
            try:
                response = generate_together(prompt, config)
                output = response['output']['choices'][0]['text'].strip()
                break
            except:
                print("Error: Retrying...")
                time.sleep(2)
                continue
        
        copied_instance = copy.deepcopy(instance)
        copied_instance['{}_generation'.format(config['task_num'])] = output
        outputs.append(copied_instance)
    
    assert len(outputs) == len(dataset)
    return outputs


def run_chat_inference(instance, config):

    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instance['prompt']}
    ]
    
    resp = call_gpt3(prompt, config['mode'], config['model']['name'], config['openai_params'])

    instance['{}_generation'.format(config['task_num'])] = resp.choices[0].message["content"]

    return instance

def run_chat(dataset, config):
    outputs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        
        for instance in tqdm(dataset, total=len(dataset)):
            copied_instance = copy.deepcopy(instance)
            
            future = executor.submit(run_chat_inference, copied_instance, config)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ret = future.result()
            outputs.append(ret)

    assert len(outputs) == len(dataset)
    return outputs


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    # set CUDA
    if isinstance(_config["gpu_id"], list):
        cuda_devices = ",".join([
            str(ele) for ele in _config["gpu_id"]
        ])
    else:
        cuda_devices = str(_config["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    console.log("GPU: {}".format(cuda_devices))

    # set seed
    fixed_seed(_config["seed"])

    result_save_dir = os.path.join(
        _config['result_save_dir'], 
        _config['file_version'],
        _config['dataset_name'], 
        _config['template_name'],
        'seed_{}'.format(_config['seed'])
    )
    if _config['task_num'] == 'task2':
        result_save_dir = os.path.join(result_save_dir, _config['task2_restriction_types'])
    if _config['task_num'] == 'augment':
        result_save_dir = os.path.join(result_save_dir, _config['datatype'])

    os.makedirs(result_save_dir, exist_ok=True)
    console.log(f"Result save directory: {result_save_dir}")

    dataset = prepare_prompt_dataset(_config)

    #with open(os.path.join('./dataset', '{}.json'.format(_config['template_name'])), 'r') as f:
    #    dataset = json.load(f)

    if _config["do_sample"]:
        dataset = dataset[:_config["sample_num"]]
        console.log("Sampling is applied for the fast debug.")

    if _config['framework'] == 'together':
        llm_outputs = run_together(dataset, _config)
    elif _config['framework'] == 'openai':
        if _config['mode'] == 'chat':
            llm_outputs = run_chat(dataset, _config)
        else:
            llm_outputs = run_completion(dataset, _config)
    
    if _config['do_cot']:
        with open(os.path.join(result_save_dir, 'cot-{}.json'.format(_config['model']['id'])), 'w') as f:
            json.dump(llm_outputs, f, ensure_ascii=False, indent='\t')

    else:
        with open(os.path.join(result_save_dir, '{}.json'.format(_config['model']['id'])), 'w') as f:
            json.dump(llm_outputs, f, ensure_ascii=False, indent='\t')
