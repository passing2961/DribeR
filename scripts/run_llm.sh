#!/bin/bash

PROMPT_NAME="new_basis_prev_history_merge_tasks"
FILE_VERSION="v3.1"
TASK_NUM="task1"


python3 open_llm_main.py with photochat_config huggingface_llm_config llama_2_chat_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="4,5" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 apply_original_prompt=True

python3 open_llm_main.py with photochat_config huggingface_llm_config vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="4,5" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 apply_original_prompt=True
