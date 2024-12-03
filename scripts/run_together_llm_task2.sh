#!/bin/bash

PROMPT_NAME="new_task2_description_qa_inst_w_rest"
FILE_VERSION="v4.0-stochastic"
TASK_NUM="task2"

'''
python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"


python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"


python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"

'''

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"


PROMPT_NAME="new_task2_description_qa_inst_w_rest_evidence"


'''
python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"


python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_llama_2_chat_70b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"


python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_vicuna_13b_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"

'''

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config together_llm_task2_config tg_falcon_40b_instruct_config  \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"
