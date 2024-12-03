#!/bin/bash


python3 open_vlm_main.py with photochat_config huggingface_llm_config blip2_opt_2_7b_config sample_debug \
    file_version="pilot" result_save_dir="./results" \
    prompt_name="basis_prev_history_task1_w_restriction:real_name" batch_size=20 sample_num=20 gpu_id="2"

python3 open_vlm_main.py with photochat_config huggingface_llm_config blip2_opt_2_7b_config sample_debug \
    file_version="pilot" result_save_dir="./results" \
    prompt_name="basis_prev_history_merge_tasks:real_name" batch_size=20 sample_num=20 gpu_id="2"