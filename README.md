# Large Language Models can Share Images, Too!

[PhotoChat++](https://huggingface.co/datasets/passing2961/photochat_plus) | [📄 Arxiv](https://arxiv.org/abs/2310.14804) | [📕 PDF](https://arxiv.org/pdf/2310.14804)

> 🚨 Disclaimer: All datasets are intended to be used for research purposes only.


## PhotoChat++

You can now load PhotoChat++ from the [HuggingFace hub](https://huggingface.co/datasets/passing2961/photochat_plus) as the following:
```python
from datasets import load_dataset

dataset = load_dataset("passing2961/photochat_plus")
```

> 🚨 Disclaimer: Since PhotoChat++ is constructed via crowd-sourcing based on dialogues from the PhotoChat dataset, which is licensed under the CC BY 4.0 International license, PhotoChat++ is shared under the same CC BY 4.0 International license. Therefore, following this license, it is possible to use the PhotoChat++ dataset for commercial purposes. However, we strongly recommend using our dataset for academic and research purposes.

## How to Run?

### Environment Variables
Set the following API keys as environment variables:
- **OPENAI_API_KEY**: Your OpenAI API key.
- **TOGETHER_API_KEY**: Your Together API key.

You can set these variables in your terminal session as follows:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export TOGETHER_API_KEY="your_together_api_key"
```

### Execute Scripts
Navigate to the `scripts` folder and execute the appropriate script for your task (e.g., ```main.py```). Below are the scripts for the evaluation:

- **Evaluation Stage 1**:  
  Run the following command:
  ```bash
  bash scripts/run_eval_task1.sh
  ```
  
- **Evaluation Stage 2**:  
  Run the following command:
  ```bash
  bash scripts/run_eval_task2.sh
  ```

- **Evaluation Stage 3**:  
  Run the following command:
  ```bash
  bash scripts/run_eval_task3.sh
  ```
  
## Citation

```
@article{lee2023large,
  title={Large Language Models can Share Images, Too!},
  author={Lee, Young-Jun and Lee, Dokyong and Sung, Joo Won and Hyeon, Jonghwan and Choi, Ho-Jin},
  journal={arXiv preprint arXiv:2310.14804},
  year={2023}
}
```
