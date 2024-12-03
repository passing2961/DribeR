from sacred import Experiment

ex = Experiment("LLM-Image-Sharing")

@ex.config
def config():
    seed = 0

    framework = None
    
    # for huggingface configures
    model = {
        #'device_map': None,
        'name': None,
        'id': None,
    }
    hf_params = {
        "max_new_tokens": None,
        #"early_stopping": False,
        #"num_beams": None,
        "do_sample": False,
        #"top_k": None,
        "top_p": None,
        #"num_return_sequences": None
    }
    skip_special_tokens = False

    mode = None
    
    batch_size = None

    openai_params = {
        "max_tokens": None,
        "temperature": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "top_p": None
    }

    together_params = {
        "max_tokens": None,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None
    }

    data_dir = None
    dataset_name = None
    datatype = None

    # prompt template
    template_name = None
    name_type = None
    do_sample_prompt = None
    num_sample_prompt = None

    result_save_dir = None
    file_version = None

    gpu_id = None
    num_gpus = None

    do_sample = False
    sample_num = None

    apply_original_prompt = False
    task_num = None

    t2i_model_name = None

    do_cot = None

    task2_restriction_types = None

    vllm_model_name = None

@ex.named_config
def sample_debug():
    do_sample = True
    sample_num = 5

@ex.named_config
def photochat_config():
    data_dir = '/home/work/workspace/LM_image_sharing/data/photochat/test.json'
    dataset_name = 'photochat'

@ex.named_config
def photochat_train_config():
    data_dir = '/home/work/workspace/LM_image_sharing/data/photochat/train.json'
    dataset_name = 'photochat'
    datatype = 'train'

@ex.named_config
def photochat_valid_config():
    data_dir = '/home/work/workspace/LM_image_sharing/data/photochat/valid.json'
    dataset_name = 'photochat'
    datatype = 'valid'

@ex.named_config
def photochat_test_config():
    data_dir = '/home/work/workspace/LM_image_sharing/data/photochat/test.json'
    dataset_name = 'photochat'
    datatype = 'test'


@ex.named_config
def photochat_plus_config():
    data_dir = './photochat++/test.json'
    dataset_name = 'photochat++'

@ex.named_config
def openai_llm_config():
    framework = "openai"
    
    openai_params = {
        "max_tokens": 1024,
        "temperature": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "top_p": 0.0
    }

@ex.named_config
def openai_llm_task2_config():
    framework = "openai"

    openai_params = {
        "max_tokens": 512,
        "temperature": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "top_p": 0.95,
    }

@ex.named_config
def openai_llm_augment_config():
    framework = "openai"
    
    openai_params = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "top_p": 0.0
    }

@ex.named_config
def together_llm_augment_config():
    framework = "together"

    together_params = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 0.0,
        "repetition_penalty": 1.0
    }

@ex.named_config
def huggingface_llm_config():
    framework = "huggingface"

#    mode = "AutoModelForCausalLM"

    skip_special_tokens = True

    hf_params = {
        "max_new_tokens": 256,
        #"early_stopping": True,
        #"num_beams": 3,
        "do_sample": True,
        #"temperature": 0.9,
        #"top_k": 50,
        #"num_return_sequences": 1,
        "top_p": 1.0
    }

@ex.named_config
def together_llm_config():
    framework = "together"

    together_params = {
        "max_tokens": 512,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 0.0,
        "repetition_penalty": 1.0
    }

@ex.named_config
def together_llm_task2_config():
    framework = "together"

    together_params = {
        "max_tokens": 256,
        "temperature": 0.9,
        "top_k": 1,
        "top_p": 0.95,
        "repetition_penalty": 1.0
    }

@ex.named_config
def vicuna_13b_config():
    model = {
        'name': "lmsys/vicuna-13b-v1.5",
        'id': "vicuna-13b"
    }
    num_gpus = 2

@ex.named_config
def tg_vicuna_13b_config():
    model = {
        'name': "lmsys/vicuna-13b-v1.5",
        'id': "together-vicuna-13b"
    }
    num_gpus = 2

@ex.named_config
def openassist_12b_config():
    model = {
        'name': "OpenAssistant/oasst-sft-1-pythia-12b",
        'id': 'oasst1-12b'
    }
    num_gpus = 2

@ex.named_config
def falcon_7b_instruct_config():
    model = {
        'name': "tiiuae/falcon-7b-instruct",
        'id': 'falcon-7b-instruct'
    }
    num_gpus = 1

@ex.named_config
def tg_falcon_7b_instruct_config():
    model = {
        'name': "togethercomputer/falcon-7b-instruct",
        'id': 'together-falcon-7b-instruct'
    }
    num_gpus = 1

@ex.named_config
def tg_falcon_40b_instruct_config():
    model = {
        'name': "togethercomputer/falcon-40b-instruct",
        'id': 'together-falcon-40b-instruct'
    }
    num_gpus = 1

@ex.named_config
def dolly_v2_12b_config():
    model = {
        'name': "databricks/dolly-v2-12b",
        'id': 'dolly-v2-12b'
    }
    num_gpus = 2
    # trust_remote_code = True

@ex.named_config
def baize_v2_13b_config():
    model = {
        'name': "project-baize/baize-v2-13b",
        'id': 'baize-v2-13b'
    }
    num_gpus = 2

@ex.named_config
def llama_2_chat_13b_config():
    model = {
        'name': "meta-llama/Llama-2-13b-chat-hf",
        'id': 'llama-2-13b-chat-hf'
    }
    num_gpus = 2

@ex.named_config
def llama_2_chat_70b_config():
    model = {
        'name': "meta-llama/Llama-2-70b-chat-hf",
        'id': 'llama-2-70b-chat-hf'
    }
    num_gpus = 8

@ex.named_config
def tg_llama_2_chat_7b_config():
    model = {
        'name': "togethercomputer/llama-2-7b-chat",
        'id': 'together-llama-2-7b-chat'
    }

@ex.named_config
def tg_llama_2_chat_13b_config():
    model = {
        'name': "togethercomputer/llama-2-13b-chat",
        'id': 'together-llama-2-13b-chat'
    }

@ex.named_config
def tg_llama_2_chat_70b_config():
    model = {
        'name': "togethercomputer/llama-2-70b-chat",
        'id': 'together-llama-2-70b-chat'
    }

@ex.named_config
def wizard_13b_config():
    model = {
        'name': "WizardLM/WizardLM-13B-V1.2",
        'id': 'wizard-13b'
    }
    num_gpus = 2

@ex.named_config
def mpt_7b_instruct_config():
    model = {
        'name': "mosaicml/mpt-7b-instruct",
        'id': 'mpt-7b-instruct'
    }
    num_gpus = 1

@ex.named_config
def mpt_30b_instruct_config():
    model = {
        'name': "mosaicml/mpt-30b-instruct",
        'id': 'mpt-30b-instruct'
    }
    num_gpus = 8

@ex.named_config
def mistral_7b_instruct_config():
    model = {
        'name': "mistralai/Mistral-7B-Instruct-v0.1",
        'id': 'mistral-7b-instruct'
    }
    num_gpus = 1

@ex.named_config
def tg_mistral_7b_instruct_config():
    model = {
        'name': "mistralai/Mistral-7B-Instruct-v0.1",
        'id': 'together-mistral-7b-instruct'
    }
    num_gpus = 1

@ex.named_config
def blip2_opt_2_7b_config():
    model = {
        'name': 'Salesforce/blip2-opt-2.7b',
        'id': 'blip2-opt-2.7b'
    }
    num_gpus = 1

@ex.named_config
def chatgpt_0613_config(): 
    # 0613
    model = {
        'name': 'gpt-3.5-turbo',
        'id': 'gpt-3.5-turbo-0613'
    }
    mode = "chat"

@ex.named_config
def chatgpt_1106_config():
    model = {
        'name': 'gpt-3.5-turbo-1106',
        'id': 'gpt-3.5-turbo-1106'
    }
    mode = "chat"

@ex.named_config
def gpt4_0613_Nov_config():
    # 0613 nov
    model = {
        'name': 'gpt-4',
        'id': 'gpt-4-0613-Nov'
    }
    mode = "chat"

@ex.named_config
def gpt4_0613_June_config():
    # 0613 nov
    model = {
        'name': 'gpt-4-0613',
        'id': 'gpt-4-0613-june'
    }
    mode = "chat"

@ex.named_config
def gpt4_0314_config():
    # 0613
    model = {
        'name': 'gpt-4-0314',
        'id': 'gpt-4-0314'
    }
    mode = "chat"

@ex.named_config
def gpt4_1106_config():
    # 0613
    model = {
        'name': 'gpt-4-1106-preview',
        'id': 'gpt-4-1106-preview'
    }
    mode = "chat"