import os
import sys
import copy
import json

import torch
import ray
from tqdm import tqdm
from rich.console import Console
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    Blip2Model,
    Blip2ForConditionalGeneration
)
from PIL import Image

from dataset import PhotoChatDataset
from config import ex
from utils import fixed_seed
from load_model import get_conversation_template

console = Console()
error_console = Console(stderr=True, style='bold red')


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def run_eval(dataset, config, result_save_dir):
    # split question file into num_gpus files
    chunk_size = len(dataset) // config['num_gpus']
    ans_handles = []
    for i in range(0, len(dataset), chunk_size):
        ans_handles.append(get_model_answers.remote(
            config['model']['name'], 
            config['model']['id'], 
            config['hf_params'],
            dataset[i:i + chunk_size])
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    return ans_jsons

@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, model_params, dataset):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(model_path,
        torch_dtype=torch.float16).cuda()

    outputs = []
    for i, instance in enumerate(tqdm(dataset)):
        
        #image_file = instance['image_file']
        #image = Image.open(image_file).convert("RGB")
        
        #inputs = processor(images=image)
        prompt = instance['prompt']

        with open(f'./actual_prompt/{model_id}.txt', 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        inputs = tokenizer([prompt])
        output_ids = model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            **model_params
        )
        output_ids = output_ids[0][len(inputs.input_ids[0]) :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        copied_instance = copy.deepcopy(instance)
        copied_instance['generation'] = output
        outputs.append(output)
    
    #assert len(outputs) == len(dataset)
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
        _config['prompt_name'],
        'seed_{}'.format(_config['seed'])
    )
    os.makedirs(result_save_dir, exist_ok=True)
    console.log(f"Result save directory: {result_save_dir}")

    #dataset = PhotoChatDataset(_config).prepare_prompt_dataset()
    with open(os.path.join('./dataset', '{}.json'.format(_config['prompt_name'])), 'r') as f:
        dataset = json.load(f)

    if _config["do_sample"]:
        dataset = dataset[:_config["sample_num"]]
        console.log("Sampling is applied for the fast debug.")

    ray.init()
    llm_outputs = run_eval(dataset, _config, result_save_dir)

    with open(os.path.join(result_save_dir, '{}.json'.format(_config['model']['id'])), 'w') as f:
        json.dump(llm_outputs, f, ensure_ascii=False, indent='\t')
