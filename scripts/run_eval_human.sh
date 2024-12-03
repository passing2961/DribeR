#!/bin/bash

FILE_VERSION="v4.0-human"
TASK_NUM="human"


python eval.py with photochat_config openai_llm_config chatgpt_0613_config \
    file_version=${FILE_VERSION} result_save_dir="./results" \
    template_name=${PROMPT_NAME} batch_size=20 sample_num=20 task_num=${TASK_NUM} \
    name_type="real_name" t2i_model_name="clip-base-16" gpu_id="4" 
