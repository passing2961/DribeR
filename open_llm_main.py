import os
import sys
import copy
import json
import time
import requests

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

from dataset import PhotoChatDataset
from config import ex
from utils import fixed_seed
from load_model import get_conversation_template
from load_dataset import prepare_prompt_dataset

console = Console()
error_console = Console(stderr=True, style='bold red')



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def generate_together(prompt, config):
    together.api_key = os.getenv("TOGETHER_API_KEY")
    #system_prompt = (
    #    "You are a helpful assistant."
    #)
    #prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\n\n" + prompt + "[/INST]"
    #prompt = f"User: {prompt} Assistant:"
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


def run_eval(dataset, config):
    # split question file into num_gpus files
    chunk_size = len(dataset) // config['num_gpus']
    ans_handles = []
    for i in range(0, len(dataset), chunk_size):
        ans_handles.append(get_model_answers.remote(
            config['model']['name'], 
            config['model']['id'], 
            config['hf_params'],
            dataset[i:i + chunk_size],
            task_num=config['task_num'],
            apply_original_prompt=config['apply_original_prompt'])
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    return ans_jsons

@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, model_params, dataset, task_num=None, apply_original_prompt=False):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    if 'oasst1' in model_id or 'mpt' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.float16).cuda()
    
    outputs = []
    for i, instance in enumerate(tqdm(dataset)):
        
        if apply_original_prompt:
            conv = get_conversation_template(model_id)
            conv.append_message(conv.roles[0], instance['prompt'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        else:
            prompt = instance['prompt']

        if apply_original_prompt:
            prompt_save_dir = os.path.join('./actual_prompt', 'original_prompt')
        else:
            prompt_save_dir = os.path.join('./actual_prompt', 'no_original_prompt')
        os.makedirs(prompt_save_dir, exist_ok=True)

        with open(os.path.join(prompt_save_dir, f'{model_id}.txt'), 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            **model_params
        )
        output_ids = output_ids[0][len(inputs.input_ids[0]) :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        copied_instance = copy.deepcopy(instance)
        copied_instance[f'{task_num}_generation'] = output
        outputs.append(copied_instance)
    
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
    os.makedirs(result_save_dir, exist_ok=True)
    console.log(f"Result save directory: {result_save_dir}")

    dataset = prepare_prompt_dataset(_config)

    #with open(os.path.join('./dataset', '{}.json'.format(_config['template_name'])), 'r') as f:
    #    dataset = json.load(f)

    if _config["do_sample"]:
        dataset = dataset[:_config["sample_num"]]
        console.log("Sampling is applied for the fast debug.")

    if 'together' in _config['model']['id']:
        llm_outputs = run_together(dataset, _config)
    else:
        ray.init()
        llm_outputs = run_eval(dataset, _config)

    if _config['do_cot']:
        with open(os.path.join(result_save_dir, 'cot-{}.json'.format(_config['model']['id'])), 'w') as f:
            json.dump(llm_outputs, f, ensure_ascii=False, indent='\t')

    else:
        with open(os.path.join(result_save_dir, '{}.json'.format(_config['model']['id'])), 'w') as f:
            json.dump(llm_outputs, f, ensure_ascii=False, indent='\t')
