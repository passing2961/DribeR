import os
import copy
import json
import glob
import random

import torch
from tqdm import tqdm
from rich.console import Console

console = Console()


TEMPLATE_DIR = './templates'

GENERIC_EXPLANATION = """The generic term refers to the visual information that is solely perceived from the given image.
For example, 
- the generic term of "Ferrari" is "Car"
- the generic term of "B. Clinton" is "Men"
- the generic term of "Burj Khalifa" is "Building"
"""

SPECIFIC_EXPLANATION = """
The specific term refers to the information that requires background knowledge, meaning it is not directly perceived from the given image.
For example,
- the specific term of "Car" can be "Ferrari"
- the specific term of "Men" can be "B. Clinton"
- the specific term of "Building" can be "Burj Khalifa"
"""

GENERIC_PROMPT = "You should describe the relevant image only using generic terms."

SPECIFIC_PROMPT = "You should describe the relevant image only using specific terms."

LESS_PROMPT = "You should describe the relevant image in less than 20 words."

MORE_PROMPT = "You should describe the relevant image in over 20 words."

CONCISE_PROMPT = "You should describe the relevant image concisely."

DETAIL_PROMPT = "You should describe the relevant image in detail."

PREFIX_PROMPT = 'You should describe the relevant image starting with "An image of".'

RESTRICTION_COLLECTIONS = {
    'generic': (GENERIC_EXPLANATION, GENERIC_PROMPT),
    'specific': (SPECIFIC_EXPLANATION, SPECIFIC_PROMPT),
    'less': ('', LESS_PROMPT),
    'more': ('', MORE_PROMPT),
    'concise': ('', CONCISE_PROMPT),
    'detail': ('', DETAIL_PROMPT),
    'prefix': ('', PREFIX_PROMPT)
}

def _load_json(data_dir):
    with open(data_dir, 'r') as f:
        return json.load(f)

def _load_SSN_name_list():
    # following SODA paper
    all_names = {}
    for dir in glob.glob(os.path.join('/home/work/workspace/iccv_codes/gpt3_test/names', '*.txt')):
        with open(dir, 'r', encoding='utf-8') as f:
            lines = [line.strip().split(',') for line in f.readlines()]
        
        for name, sex, count in lines:
            all_names[name] = [int(count), sex]

    sorted_names = sorted(all_names.items(), key=lambda x: x[1][0], reverse=True)[:1000]
    sex = [ele[1][1] for ele in sorted_names]
    
    #print('# of Female names: {}'.format(sex.count('F')))
    #print('# of Male names: {}'.format(sex.count('M')))

    return sorted_names

def _load_template(template_name):
    with open(os.path.join(TEMPLATE_DIR, f'{template_name}.txt'), 'r') as f:
        return f.read()

def _load_sampled_dataset_for_task1(dataset, num_sample):
    results = []

    for i, instance in enumerate(tqdm(dataset, total=len(dataset))):
        dialog = instance['dialogue']

        _context, _speaker = [], []
        for j, ele in enumerate(dialog):
            if ele['share_photo']:
                share_turn_idx = j
                break
            
            _context.append(ele['message'])
            _speaker.append(ele['user_id'])

        share_turn_speaker = dialog[share_turn_idx]['user_id']

        prev_len = [e for e in range(1, len(_context))]
        print(len(prev_len), _context)
        sampled_indices = random.sample(prev_len, num_sample)

        for index in sampled_indices:
            neg_context, neg_speaker = _context[:index], _speaker[:index]
            assert len(neg_context) > 0

            copied_instance = copy.deepcopy(instance)
            copied_instance['image_file_dir'] = copied_instance['image_file_dir']
            copied_instance['share_turn_idx'] = share_turn_idx
            copied_instance['share_turn_speaker'] = _speaker[index]
            copied_instance['context'] = neg_context
            copied_instance['speaker'] = neg_speaker
            copied_instance['full_context'] = [ele['message'] for ele in dialog]
            copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
            copied_instance['label'] = 'no'

            results.append(copied_instance)
        
        copied_instance = copy.deepcopy(instance)
        copied_instance['image_file_dir'] = copied_instance['image_file_dir']
        copied_instance['share_turn_idx'] = share_turn_idx
        copied_instance['share_turn_speaker'] = share_turn_speaker
        copied_instance['context'] = _context
        copied_instance['speaker'] = _speaker
        copied_instance['full_context'] = [ele['message'] for ele in dialog]
        copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
        copied_instance['label'] = 'yes'
    
        results.append(copied_instance)
    return results

def _load_dataset_for_gt_task1(dataset):
    results = []

    for i, instance in enumerate(tqdm(dataset, total=len(dataset))):
        dialog = instance['dialogue']

        _context, _speaker = [], []
        for j, ele in enumerate(dialog):
            if ele['share_photo']:
                share_turn_idx = j
                break
            
            _context.append(ele['message'])
            _speaker.append(ele['user_id'])

        share_turn_speaker = dialog[share_turn_idx]['user_id']
        
        copied_instance = copy.deepcopy(instance)
        copied_instance['image_file_dir'] = copied_instance['image_file_dir']
        copied_instance['share_turn_idx'] = share_turn_idx
        copied_instance['share_turn_speaker'] = share_turn_speaker
        copied_instance['context'] = _context
        copied_instance['speaker'] = _speaker
        copied_instance['full_context'] = [ele['message'] for ele in dialog]
        copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
        copied_instance['label'] = 'yes'
    
        results.append(copied_instance)
    return results

def _load_dataset_for_task1(dataset):
    results = []

    for i, instance in enumerate(tqdm(dataset, total=len(dataset))):
        dialog = instance['dialogue']

        _context, _speaker = [], []
        for j, ele in enumerate(dialog):
            if ele['share_photo']:
                share_turn_idx = j
                break
            
            _context.append(ele['message'])
            _speaker.append(ele['user_id'])

        share_turn_speaker = dialog[share_turn_idx]['user_id']

        for index in range(len(_context)):
            if index == 0:
                continue
            else:
                neg_context, neg_speaker = _context[:index], _speaker[:index]
            assert len(neg_context) > 0

            copied_instance = copy.deepcopy(instance)
            copied_instance['image_file_dir'] = copied_instance['image_file_dir']
            copied_instance['share_turn_idx'] = share_turn_idx
            copied_instance['share_turn_speaker'] = _speaker[index]
            copied_instance['context'] = neg_context
            copied_instance['speaker'] = neg_speaker
            copied_instance['full_context'] = [ele['message'] for ele in dialog]
            copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
            copied_instance['label'] = 'no'

            results.append(copied_instance)
        
        copied_instance = copy.deepcopy(instance)
        copied_instance['image_file_dir'] = copied_instance['image_file_dir']
        copied_instance['share_turn_idx'] = share_turn_idx
        copied_instance['share_turn_speaker'] = share_turn_speaker
        copied_instance['context'] = _context
        copied_instance['speaker'] = _speaker
        copied_instance['full_context'] = [ele['message'] for ele in dialog]
        copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
        copied_instance['label'] = 'yes'
    
        results.append(copied_instance)
    return results


def _load_dataset_for_augment(dataset):
    results = []

    for i, instance in enumerate(tqdm(dataset, total=len(dataset))):
        dialog = instance['dialogue']

        _context, _speaker = [], []
        for j, ele in enumerate(dialog):
            if ele['share_photo']:
                share_turn_idx = j
                continue
            
            _context.append(ele['message'])
            _speaker.append(ele['user_id'])

        share_turn_speaker = dialog[share_turn_idx]['user_id']

        copied_instance = copy.deepcopy(instance)
        copied_instance['image_file_dir'] = copied_instance['image_file_dir']
        copied_instance['share_turn_idx'] = share_turn_idx
        copied_instance['share_turn_speaker'] = share_turn_speaker
        copied_instance['context'] = _context
        copied_instance['speaker'] = _speaker
        copied_instance['full_context'] = [ele['message'] for ele in dialog]
        copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
        
        results.append(copied_instance)
    return results

def _load_name_dict(name_type, dialog_indices):
    name_dict = dict()

    if name_type == 'real_name':
        name_list = _load_SSN_name_list()
        
        for index in dialog_indices:
            sampled_names = {i: ele for i, ele in enumerate(random.sample(name_list, 2))}
            name_dict[index] = sampled_names
        return name_dict
    elif name_type == 'ab_name':
        for index in dialog_indices:
            name_dict[index] = {0: ['Speaker A'], 1: ['Speaker B']}
        return name_dict

def prepare_task1_dataset(config):
    original_dataset = _load_json(config['data_dir'])

    if config['do_sample_prompt']:
        dataset = _load_sampled_dataset_for_task1(original_dataset, config['num_sample_prompt'])
    else:
        if config['file_version'] == 'only_gt_task1':
            dataset = _load_dataset_for_gt_task1(original_dataset)
        else:
            dataset = _load_dataset_for_task1(original_dataset)

    dialog_idx_list = list(set([ele['dialogue_id'] for ele in dataset]))

    template = _load_template(config['template_name'])
    name_dict = _load_name_dict(config['name_type'], dialog_idx_list)

    prompts = []
    for instance in tqdm(dataset, total=len(dataset)):
        context, speaker = instance['context'], instance['speaker']
        share_speaker = instance['share_turn_speaker']
        dialog_idx = instance['dialogue_id']

        flatten_dialog = []
        for i, (utt, spk) in enumerate(zip(context, speaker)):
            flatten_dialog += [
                "{}: {}".format(
                    name_dict[dialog_idx][spk][0], utt
                )
            ]

        flatten_dialog = '\n'.join(flatten_dialog)
        prompt = template.format(
            dialogue=flatten_dialog,
            spk1=name_dict[dialog_idx][0][0],
            spk2=name_dict[dialog_idx][1][0],
            share_spk=name_dict[dialog_idx][share_speaker][0]
        )
        
        if config['do_cot']:
            prompt += " Let's think step by step."


        copied_instance = copy.deepcopy(instance)
        copied_instance['prompt'] = prompt
        copied_instance['share_turn_speaker_name'] = name_dict[dialog_idx][share_speaker][0]
        copied_instance['speaker_mapping_table'] = {0: name_dict[dialog_idx][0][0], 1: name_dict[dialog_idx][1][0]}

        prompts.append(copied_instance)

    console.log('Total number of prompts: {}'.format(len(prompts)))

    return prompts

def _load_dataset_for_task2(dataset):
    results = []

    for i, instance in enumerate(tqdm(dataset, total=len(dataset))):
        dialog = instance['dialogue']

        context, speaker = [], []
        for j, ele in enumerate(dialog):
            if ele['share_photo']:
                share_turn_idx = j
                break
            
            context.append(ele['message'])
            speaker.append(ele['user_id'])
        
        share_turn_speaker = dialog[share_turn_idx]['user_id']

        copied_instance = copy.deepcopy(instance)
        copied_instance['image_file_dir'] = copied_instance['image_file_dir']
        copied_instance['share_turn_idx'] = share_turn_idx
        copied_instance['share_turn_speaker'] = share_turn_speaker
        copied_instance['context'] = context
        copied_instance['speaker'] = speaker
        copied_instance['full_context'] = [ele['message'] for ele in dialog]
        copied_instance['full_speaker'] = [ele['user_id'] for ele in dialog],
        copied_instance['label'] = 'yes'

        results.append(copied_instance)
    
    return results

def construct_restriction_prompt(config):

    restriction_types = config['task2_restriction_types'].split(',')

    restriction_prompt = []
    restriction_explanation = []
    for i, ret_type in enumerate(restriction_types):
        _restriction_explanation, restriction_sentence = RESTRICTION_COLLECTIONS[ret_type]
        if 'evidence' in config['template_name']:
            restriction_prompt += [
                f'({i+3}) {restriction_sentence}'
            ]
        else:
            restriction_prompt += [
                f'({i+2}) {restriction_sentence}'
            ]
        if _restriction_explanation:
            restriction_explanation.append(_restriction_explanation)

    return '\n'.join(restriction_prompt), restriction_explanation

def prepare_task2_dataset(config):
    original_dataset = _load_json(config['data_dir'])

    dataset = _load_dataset_for_task2(original_dataset)
    dialog_idx_list = list(set([ele['dialogue_id'] for ele in dataset]))

    template = _load_template(config['template_name'])
    name_dict = _load_name_dict(config['name_type'], dialog_idx_list)

    restriction_prompt, restriction_explanation = construct_restriction_prompt(config)

    prompts = []
    for instance in tqdm(dataset, total=len(dataset)):
        context = instance['context']
        speaker = instance['speaker']
        share_speaker = instance['share_turn_speaker']
        dialog_idx = instance['dialogue_id']

        flatten_dialog = []
        for i, (utt, spk) in enumerate(zip(context, speaker)):
            flatten_dialog += [
                "{}: {}".format(
                    name_dict[dialog_idx][spk][0], utt
                )
            ]
        
        flatten_dialog += [
            "{}: [Sharing Image]".format(name_dict[dialog_idx][share_speaker][0])
        ]
        flatten_dialog = '\n'.join(flatten_dialog)

        prompt = template.format(
            dialogue=flatten_dialog,
            spk1=name_dict[dialog_idx][0][0],
            spk2=name_dict[dialog_idx][1][0],
            share_spk=name_dict[dialog_idx][share_speaker][0],
            restriction=restriction_prompt
        )
        if restriction_explanation:
            position = prompt.find("Dialogue:")
            prompt = prompt[:position] + restriction_explanation[0] + '\n' + prompt[position:]

        copied_instance = copy.deepcopy(instance)
        copied_instance['prompt'] = prompt
        copied_instance['share_turn_speaker_name'] = name_dict[dialog_idx][share_speaker][0]
        copied_instance['speaker_mapping_table'] = {0: name_dict[dialog_idx][0][0], 1: name_dict[dialog_idx][1][0]}

        prompts.append(copied_instance)
    
    console.log('Total number of prompts: {}'.format(len(prompts)))

    return prompts


def prepare_augment_dataset(config):
    original_dataset = _load_json(config['data_dir'])

    dataset = _load_dataset_for_augment(original_dataset)

    dialog_idx_list = list(set([ele['dialogue_id'] for ele in dataset]))

    template = _load_template(config['template_name'])
    name_dict = _load_name_dict(config['name_type'], dialog_idx_list)

    prompts = []
    for instance in tqdm(dataset, total=len(dataset)):
        context, speaker = instance['context'], instance['speaker']
        share_speaker = instance['share_turn_speaker']
        dialog_idx = instance['dialogue_id']

        flatten_dialog = []
        for i, (utt, spk) in enumerate(zip(context, speaker)):
            flatten_dialog += [
                "{}: {}".format(
                    name_dict[dialog_idx][spk][0], utt
                )
            ]

        flatten_dialog = '\n'.join(flatten_dialog)
        prompt = template.format(
            dialogue=flatten_dialog,
            spk1=name_dict[dialog_idx][0][0],
            spk2=name_dict[dialog_idx][1][0],
        )

        copied_instance = copy.deepcopy(instance)
        copied_instance['prompt'] = prompt
        copied_instance['share_turn_speaker_name'] = name_dict[dialog_idx][share_speaker][0]
        copied_instance['speaker_mapping_table'] = {0: name_dict[dialog_idx][0][0], 1: name_dict[dialog_idx][1][0]}

        prompts.append(copied_instance)

    console.log('Total number of prompts: {}'.format(len(prompts)))

    return prompts

def prepare_prompt_dataset(config):

    task_num = config['task_num']

    if task_num == 'task1':
        return prepare_task1_dataset(config)
    elif task_num == 'task2':
        return prepare_task2_dataset(config)
    elif task_num == 'augment':
        return prepare_augment_dataset(config)
    else:
        raise ValueError("Wrong task_num!")