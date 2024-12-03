#!/bin/bash

PROMPT_NAME="new_task2_description_qa_inst_w_rest"
FILE_VERSION="v1.0"
TASK_NUM="task2"

python eval.py with photochat_config together_llm_config tg_vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix"

PROMPT_NAME="new_task2_description_qa_inst_w_rest_evidence"

python eval.py with photochat_config together_llm_config tg_vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix"

PROMPT_NAME="new_task2_description_qa_inst_w_rest"

python eval.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" \
    task2_restriction_types="prefix"
