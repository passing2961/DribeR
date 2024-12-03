#!/bin/bash

PROMPT_NAME="new_task2_description_qa_inst_w_rest"
FILE_VERSION="v4.0-stochastic"
TASK_NUM="vllm"

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix" vllm_model_name="Qwen-VL-Chat"

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix" vllm_model_name="Flamingo"

'''
python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix" vllm_model_name="LLaVAv1.5-13B"

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix" vllm_model_name="MiniGPT4_13B"

python eval.py with photochat_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix" vllm_model_name="MiniGPT4"

'''