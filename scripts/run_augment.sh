#!/bin/bash

PROMPT_NAME="new_all_generation"
FILE_VERSION="v4.0-augment"
TASK_NUM="augment"

python main.py with photochat_train_config together_llm_augment_config tg_vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" 
