import os
import random
import numpy as np
import torch


def fixed_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


length_prompts = {
    'less': "You should describe the relevant image in less than 20 words.",
    'more': "You should describe the relevant image in over 20 words."
}

situation_prompts = {
    'concise': "You should describe the relevant image concisely",
    'detail': "You should describe the relevant image in detail",
}

generic_specific_prompts = {
    'generic': "You should describe the relevant image only using the generic terms. (e.g., convert 'Ferrari' to 'car')",
    'specific': "You should describe the relevant image using the specific terms. (e.g., convert 'car' to 'Ferrari')"
}