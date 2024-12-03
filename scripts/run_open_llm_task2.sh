#!/bin/bash

TEMPLATE_NAME="caption_task2"
FILE_VERSION="v3.1"
TASK_NUM="task2"


python3 open_llm_main.py with photochat_config huggingface_llm_config llama_2_chat_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${TEMPLATE_NAME} batch_size=20 sample_num=20 gpu_id="6,7" task_num=${TASK_NUM} \
    name_type="real_name" apply_original_prompt=True

python3 open_llm_main.py with photochat_config huggingface_llm_config vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${TEMPLATE_NAME} batch_size=20 sample_num=20 gpu_id="6,7" task_num=${TASK_NUM} \
    name_type="real_name" apply_original_prompt=True


#python3 open_llm_main.py with photochat_config huggingface_llm_config llama_2_chat_13b_config sample_debug \
#    file_version=${FILE_VERSION} result_save_dir="./results" \
#    prompt_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="6,7" task_num=${TASK_NUM} \
#    name_type="real_name"