#!/bin/bash


PROMPT_NAME="few_2_new_task1_binary_inst"
FILE_VERSION="v4.0"
TASK_NUM="task1"



python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=False num_sample_prompt=1 do_cot=False

PROMPT_NAME="few_4_new_task1_binary_inst"

python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=False num_sample_prompt=1 do_cot=False

PROMPT_NAME="few_8_new_task1_binary_inst"

python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=False num_sample_prompt=1 do_cot=False

'''

python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True


PROMPT_NAME="new_merge_multiple_choice_inst_w_sentence_evidence"

python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True


PROMPT_NAME="new_merge_multiple_choice_inst_w_sentence_word_evidence"


python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_task2_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True

python main.py with photochat_config openai_llm_task2_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 do_cot=True

PROMPT_NAME="new_task1_multiple_choice_inst_w_evidence_wo_restriction"
FILE_VERSION="v4.0-2K-wo-restriction"


python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

python main.py with photochat_config openai_llm_config gpt4_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" do_sample_prompt=True num_sample_prompt=1 #do_cot=True

'''

'''
python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"


PROMPT_NAME="new_task2_description_qa_inst_w_rest_evidence"

python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,generic"

python main.py with photochat_config openai_llm_config chatgpt_1106_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 gpu_id="0,1" task_num=${TASK_NUM} \
    name_type="real_name" task2_restriction_types="prefix,specific"

'''