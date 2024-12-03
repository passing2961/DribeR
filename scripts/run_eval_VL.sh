#!/bin/bash

PROMPT_NAME="clip-zero-shot"
FILE_VERSION="v1.0"
TASK_NUM="vl"

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="4" task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" 

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="4" task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-32" 

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="4" task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-large-14" 

