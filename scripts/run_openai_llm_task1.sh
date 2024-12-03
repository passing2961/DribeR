#!/bin/bash

PROMPT_NAME="few_1_new_task1_binary_inst"
FILE_VERSION="v4.0"
TASK_NUM="task1"



python openai_llm_main.py with photochat_train_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=False num_sample_prompt=1 do_cot=False

#PROMPT_NAME="few_2_new_task1_binary_inst"


'''
PROMPT_NAME="new_task1_multiple_choice_inst_w_evidence"

python openai_llm_main.py with photochat_config openai_llm_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True
'''