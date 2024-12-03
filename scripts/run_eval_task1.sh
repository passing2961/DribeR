#!/bin/bash

PROMPT_NAME="new_task1_binary_inst"
FILE_VERSION="v1.0"
TASK_NUM="task1"

python3 eval.py with photochat_config together_llm_config tg_vicuna_13b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM}

python3 eval.py with photochat_config together_llm_config tg_llama_2_chat_70b_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM}

python3 eval.py with photochat_config openai_llm_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM}


PROMPT_NAME="few_4_new_task1_binary_inst"

python3 eval.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM} do_cot=False

PROMPT_NAME="few_8_new_task1_binary_inst"

python3 eval.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM} do_cot=False


python3 eval.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM} do_cot=True

python3 eval.py with photochat_config openai_llm_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM} do_cot=True

PROMPT_NAME="new_merge_multiple_choice_inst_w_sentence_evidence"

python3 eval.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM}

python3 eval.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM} do_cot=True



#PROMPT_NAME="new_task1_multiple_choice_inst_w_evidence"

python3 eval.py with photochat_config together_llm_config tg_mistral_7b_instruct_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} task_num=${TASK_NUM}


