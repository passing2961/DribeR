#!/bin/bash

PROMPT_NAME="new_task1_multiple_choice_inst"
FILE_VERSION="v4.0-system-2K"
TASK_NUM="task1"



PROMPT_NAME="new_task1_multiple_choice_inst_w_evidence"


python main.py with photochat_config together_llm_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 

python main.py with photochat_config together_llm_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 


python main.py with photochat_config together_llm_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True