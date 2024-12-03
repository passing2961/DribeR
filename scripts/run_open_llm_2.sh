#!/bin/bash

PROMPT_NAME="new_basis_prev_history_merge_tasks"
FILE_VERSION="v3.3"
TASK_NUM="task1"


python open_llm_main.py with photochat_config huggingface_llm_config llama_2_chat_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 apply_original_prompt=True

#python3 open_llm_main.py with photochat_config huggingface_llm_config mistral_7b_instruct_config  \
#    file_version=${FILE_VERSION} result_save_dir="./results" \
#    prompt_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0" task_num=${TASK_NUM}


#python3 open_llm_main.py with photochat_config huggingface_llm_config vicuna_13b_config \
#    file_version="pilot" result_save_dir="./results" \
#    prompt_name="basis_prev_history_merge_tasks:real_name" batch_size=20 sample_num=20 gpu_id="0,1"

