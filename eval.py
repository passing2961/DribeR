from typing import Dict
import os
import re
import sys
import json
import time
import copy
from pathlib import Path
from collections import Counter

import ast
import concurrent.futures
import evaluate
import torch
import pandas as pd
import openai
from tqdm import tqdm
from PIL import Image
from rich.console import Console
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryConfusionMatrix
)
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPModel
)
import ImageReward as RM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import ex

console = Console()

PROJECT_HOME = Path(__file__).parent.resolve()
EVAL_DIR_PATH = os.path.join(PROJECT_HOME, 'reports')

OBJECT_TEMPLATE = """You will be provided a list of object categories and an image description. Your job is to detect the object in the image description and categorize the detected object into one of the categories in the list. 

You must provide your answer in a Python dictionary object that has the category as the key and the corresponding object in the image description as the value.

Object Category List = [
    "Woman", "Man", "Girl", "Boy", "Human body", "Face", "Bagel", "Baked goods", "Beer", "Bread", "Burrito", 
    "Cake", "Candy", "Cheese", "Cocktail", "Coffee", "Cookie", "Croissant", "Dessert", "Doughnut", "Drink", 
    "Fast food", "French fries", "Hamburger", "Hot dog", "Ice cream", "Juice", "Milk", "Pancake", "Pasta", 
    "Pizza", "Popcorn", "Salad", "Sandwich", "Seafood", "Snack", "Animal", "Alarm clock", "Backpack", "Blender", 
    "Banjo", "Bed", "Belt", "Computer keyboard", "Computer mouse", "Curtain", "Guitar", "Hair dryer", "Hair spray", 
    "Harmonica", "Humidifier", "Jacket", "Jeans", "Dress", "Earrings", "Necklace", "Fashion accessory", "Bicycle", 
    "Calculator", "Camera", "Food processor", "Jug", "Mixing bowl", "Nightstand", "Oboe", "Oven", "Paper cutter", 
    "Pencil case", "Perfume", "Pillow", "Personal care", "Pizza cutter", "Pressure cooker", "Printer", "Refridgerator", 
    "High heels", "Skateboard", "Slow cooker", "Teddy bear", "Teapot", "Vase", "Wall clock", "Taco", "Tart", "Tea", 
    "Waffle", "Wine", "Guacamole"
]

Image Description: {llm_description}

Answer:"""


def call_chatgpt(prompt_input):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    
    openai.organization = os.getenv("OPENAI_ORGANIZATION_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    while not received:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106", 
                messages=[
                    {"role": "system", "content": "You are a succinct and helpful assistant."},
                    {"role": "user", "content": prompt_input}
                ], 
                max_tokens=1024,
                temperature=0.,
                top_p=0.,
                frequency_penalty=0.,
                presence_penalty=0.
            )
           
            received = True
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g., prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{inputs}\n\n")
                assert False
            
            time.sleep(2)
    
    return response


class BaseEvalAgent():

    def __init__(self, config: Dict) -> None:
        self.config = config

        
        result_save_dir = os.path.join(
            config['result_save_dir'],
            config['file_version'],
            config['dataset_name'],
            config['template_name'],
            'seed_{}'.format(config['seed']),
        )
        if config['task2_restriction_types']:
            result_save_dir = os.path.join(result_save_dir, config['task2_restriction_types'])
        

        self.output_dir = os.path.join(
            EVAL_DIR_PATH,
            config['task_num'],
            config['file_version'],
            config['dataset_name'],
            config['template_name'],
            'seed_{}'.format(config['seed']),
        )
        if config['task2_restriction_types']:
            self.output_dir = os.path.join(self.output_dir, config['task2_restriction_types'])

        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_photochat_plus()

        if config['task_num'] == 'human':
            self.output_filename_suffix = '_human.json'
        #elif config['task_num'] == 'vllm':
        #    self.output_filename_suffix = '_{}.json'.format(config['vllm_model_name'])
        else:
            if config['t2i_model_name'] and config['task_num'] == 'vl':
                self.output_filename_suffix = '_{}.json'.format(config['t2i_model_name'])
                self.outputs = self.load_json(config['data_dir'])
            elif config['task_num'] == 'vllm':
                self.output_filename_suffix = '{}_{}.json'.format(config['vllm_model_name'], config['t2i_model_name'])
                self.outputs = self.load_json(config['data_dir'])
            else:
                if config['do_cot'] == True:
                    self.output_filename_suffix = '_cot_{}.json'.format(config['model']['id'])
                    self.outputs = self.load_json(os.path.join(result_save_dir, 'cot-{}.json'.format(config['model']['id'])))
                else:
                    self.output_filename_suffix = '_{}.json'.format(config['model']['id'])
                    self.outputs = self.load_json(os.path.join(result_save_dir, '{}.json'.format(config['model']['id'])))

    def load_json(self, datadir: str):
        with open(datadir, 'r') as f:
            return json.load(f)

    def load_photochat_plus(self):
        photochat_plus = self.load_json('./photochat++/test.json')

        self.photochat_plus = {}
        for instance in photochat_plus:
            self.photochat_plus[instance['dialogue_id']] = instance
    
    def dump_report_outputs(self, reports, evaluation_outputs):
        """
        Dump the reports and the evaluation outputs
        """
        evaluated_response_filename = "evaluated_responses" + self.output_filename_suffix
        report_filename = "reports" + self.output_filename_suffix
        
        with open(os.path.join(self.output_dir, evaluated_response_filename), 'w') as f:
            json.dump(evaluation_outputs, f, ensure_ascii=False, indent='\t')

        with open(os.path.join(self.output_dir, report_filename), 'w') as f:
            json.dump(reports, f, ensure_ascii=False, indent='\t')
        
        console.log(">>>>> Dumped evaluation outputs and the report at {}!".format(self.output_dir))
        console.log(">>>>> Evaluated model responses filename: {}".format(evaluated_response_filename))
        console.log(">>>>> REPORT filename: {}".format(report_filename))

    def map_binary_answer_to_int(self, model_response):
        """
        Maps a binary answer to an integer value.

        Args:
            model_response (str): The model's response.

        Returns:
            int: The mapped integer value. Returns 1 for positive answers (e.g., 'yes', 'true'), 
                 0 for negative answers (e.g., 'no'), and -1 for other cases.
        """
        if model_response == False:
            model_response = 'No'
        
        model_answer = model_response.lower() #.strip("'")

        if " yes," in model_answer or " yes " in model_answer or model_answer.startswith("yes") or " yes." in model_answer or 'it is appropriate' in model_answer or 'it would be appropriate' in model_answer or 'it may be appropriate' in model_answer or 'it seems appropriate' in model_answer:
            return 1
        elif " no," in model_answer or " no " in model_answer or model_answer.startswith("no") or " no." in model_answer or 'it is not appropriate' in model_answer or 'False' in model_answer or 'it may not be' in model_answer or 'it would not be' in model_answer or 'it does not seem appropriate' in model_answer or 'should not share' in model_answer:
            return 0
        else:
            return -1
        
    def map_multiple_answer_to_int(self, model_responses):
        """
        Maps multiple-choice answers to integer values.
        """
        output = []
        for model_response in model_responses:
            if isinstance(model_response, tuple):
                #pred_label = -1
                continue
            else:    
                model_answer = model_response.lower()
                if "information dissemination" in model_answer or "(a)" in model_answer or "a" == model_answer:
                    pred_label = 0
                elif "social bonding" in model_answer or "(b)" in model_answer or "b" == model_answer:
                    pred_label = 1
                elif "humor and entertainment" in model_answer or "(c)" in model_answer or "c" == model_answer:
                    pred_label = 2
                elif "visual clarification" in model_answer or "(d)" in model_answer or "d" == model_answer:
                    pred_label = 3
                elif "topic transition" in model_answer or "(e)" in model_answer or "e" == model_answer:
                    pred_label = 4
                elif "expression of emotion or opinion" in model_answer or "(f)" in model_answer or "f" == model_answer:
                    pred_label = 5
                else:
                    #pred_label = -1
                    continue

            output.append(pred_label)
        return output

    def yesno_to_int(self, yesno_str):
        """
        This is for evaluating a task1 on PhotoChat.
        """
        mapping = {'yes': 1, 'no': 0}
        return mapping[yesno_str]
    
    def intent_to_int(self, intent_str):
        """
        This is for evaluating a multiple-choice setting task1 on PhotoChat++.
        """
        if intent_str == 'human and entertainment':
            intent_str = 'humor and entertainment'

        mapping = {
            "information dissemination": 0,
            "social bonding": 1,
            "humor and entertainment": 2,
            "visual clarification": 3,
            "topic transition": 4,
            "expression of emotion or opinion": 5,
        }
        try:
            return mapping[intent_str]
        except KeyError:
            return -1

    def score_and_analyze(self, df):
        """
        Aggregate scores and performs analysis on the model responses and evaluation results.
        """
        raise NotImplementedError

    def evaluate_response(self):
        """
        Evaluates the model's response.
        """
        raise NotImplementedError

    def run_reports(self, results):
        """
        Create report after scoring and analyzing the results.

        Input:
        - results: a list of results from LLM

        Output:
        - report: a dictionary of scores and analysis
        """
        df = pd.DataFrame(results)

        report = self.score_and_analyze(df)

        return report

    def parse_llm_description(self, llm_description: str):
        if "The most appropriate image description" in llm_description:
            pattern = r'"(.*)"'

            match = re.search(pattern, llm_description, re.DOTALL)

            resp =  match.group(1) if match else None
            if not resp:
                _pattern = r"The most appropriate image description to share in the \[Sharing Image\] turn is (.*)."
                _match = re.search(_pattern, llm_description)
                _resp = _match.group(1) if _match else None
                if not _resp:
                    _resp2 = llm_description.split('\n\n')[-1]
                    assert _resp2.startswith("An image of")

                    return _resp2
                return _resp
                
            return resp
        else:
            # An image of a wine glass with a red liquid inside.
            #
            # Explanation:
            # The generic term for "wine glass" is "Container".
            # The generic term for "red liquid" is "Liquid".
            try:
                return llm_description.strip().split('\n')[0]
            except:
                return None
    
    def compute_f1(self, model_response, ground_truth):
        """
        Compute the F1 score between the ground truth and model response.

        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.

        Returns:
            float: The F1 score.
        """
        if not model_response:
            return 0

        ground_truth = ground_truth.split()
        model_response = model_response.split()
        common = Counter(ground_truth) & Counter(model_response)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(model_response)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_intent_f1(self, model_response, ground_truth):
        """
        Compute the F1 score between the ground truth and model response.

        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.

        Returns:
            float: The F1 score.
        """
        #ground_truth = ground_truth.split()
        #model_response = model_response.split()

        if len(model_response) == 0:
            return 0, 0, 0

        common = Counter(ground_truth) & Counter(model_response)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(model_response)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def compute_dist(self, model_answer, ground_truth):
        if not model_answer:
            return 0.
        pred_emb = self.embedder.encode(model_answer)
        gt_emb = self.embedder.encode(ground_truth)

        similarity = cosine_similarity(pred_emb.reshape(1, -1), gt_emb.reshape(1, -1))[0][0]

        return similarity

    def run(self):
        evaluated_outputs = self.evaluate_response(self.outputs)
        reports = self.run_reports(evaluated_outputs)
        self.dump_report_outputs(reports, evaluated_outputs)

class Task1EvalAgent(BaseEvalAgent):

    def __init__(self, config: Dict) -> None:
        config['task_num'] = 'task1'

        super().__init__(config=config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)

    def evaluate_binary_answer(self, output):
        pred_resp = output['task1_generation']
        
        binary_answer = self.map_binary_answer_to_int(pred_resp)

        if self.yesno_to_int(output['label'].lower()) == binary_answer:
            result = True
        else:
            result = False
        copied_output = copy.deepcopy(output)
        copied_output['task1_binary_prediction'] = binary_answer
        copied_output['task1_binary_correct'] = self.yesno_to_int(output['label'].lower())

        copied_output['task1_binarized_model_answer'] = binary_answer
        copied_output['task1_binary_result'] = result

        copied_output['task1_history_len'] = len(output['context'])
 
        return copied_output

    def evaluate_multiple_choice_answer(self, output):
        pred_resp = output['task1_generation']

        if "```python" in pred_resp:
            pred_resp = pred_resp.replace("```python", "").replace("```", "")
        if "answer = " in pred_resp:
            pred_resp = pred_resp.replace("answer = ", "")
        if 'null' in pred_resp:
            pred_resp = pred_resp.replace('null', 'None')
        
        if self.config['framework'] == 'together':
            if '{' in pred_resp and '}' in pred_resp:
                start_index = pred_resp.index('{')
                end_index = pred_resp.index('}')
                pred_resp = pred_resp[start_index:end_index+1]

        if self.config['do_cot']:
            try:
                start_index = pred_resp.index('{')
                end_index = pred_resp.index('}')
                pred_resp = pred_resp[start_index:end_index+1]
            except ValueError:
                pred_resp = None

        try:
            pred_dict = ast.literal_eval(pred_resp)
        except:
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': []
            }

        if not isinstance(pred_dict, dict):
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': []
            }

        if 'Prediction' not in pred_dict.keys():
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': []
            }
        if 'Intent' not in pred_dict.keys():
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': []
            }

        yesno_pred = pred_dict["Prediction"]
        intent_pred = pred_dict["Intent"]

        binary_answer = self.map_binary_answer_to_int(yesno_pred)
        multiple_answer = self.map_multiple_answer_to_int(intent_pred)

        if self.yesno_to_int(output['label'].lower()) == binary_answer:
            result = True
        else:
            result = False

        copied_output = copy.deepcopy(output)
        copied_output['task1_binary_prediction'] = binary_answer
        copied_output['task1_binary_correct'] = self.yesno_to_int(output['label'].lower())
        copied_output['task1_binarized_model_answer'] = binary_answer
        copied_output['task1_binary_result'] = result


        photochat_plus_instance = self.photochat_plus[output['dialogue_id']]
        all_intent = photochat_plus_instance['all_info:intent']
        try:
            HA_intent = photochat_plus_instance['aggrement_info:intent']
        except KeyError:
            print(photochat_plus_instance.keys())
            assert False
        assert len(all_intent) > 0
        
        choice_answers = [self.intent_to_int(_intent.lower()) for _intent in all_intent]
        copied_output['task1_multiple_prediction'] = multiple_answer
        copied_output['task1_multiple_correct'] = choice_answers
        if len(multiple_answer) == 0:

            copied_output['task1_multiple_prediction_flag'] = False
        else:
            copied_output['task1_multiple_prediction_flag'] = True

        #intersection_ratio = set(multiple_answer).intersection(set(choice_answers))
        #intersection_score = len(intersection_ratio)/len(choice_answers)
        intent_f1, intent_precision, intent_recall = self.compute_intent_f1(multiple_answer, choice_answers)
        copied_output['task1_intent_f1'] = intent_f1
        copied_output['task1_intent_precision'] = intent_precision
        copied_output['task1_intent_recall'] = intent_recall

        if len(HA_intent) == 0:
            copied_output['task1_HA_intent_f1'] = 0
            copied_output['task1_HA_intent_precision'] = 0
            copied_output['task1_HA_intent_recall'] = 0
            copied_output['task1_HA_multiple_correct'] = []
        else:
            HA_choice_answers = [self.intent_to_int(_intent.lower()) for _intent in HA_intent]
            HA_intent_f1, HA_intent_precision, HA_intent_recall = self.compute_intent_f1(multiple_answer, HA_choice_answers)
            copied_output['task1_HA_intent_f1'] = HA_intent_f1
            copied_output['task1_HA_intent_precision'] = HA_intent_precision
            copied_output['task1_HA_intent_recall'] = HA_intent_recall
            copied_output['task1_HA_multiple_correct'] = HA_choice_answers

        return copied_output

    def evaluate_evidence_answer(self, output):
        pred_resp = output['task1_generation']

        if "```python" in pred_resp:
            pred_resp = pred_resp.replace("```python", "").replace("```", "")
        if "answer = " in pred_resp:
            pred_resp = pred_resp.replace("answer = ", "")
        if 'null' in pred_resp:
            pred_resp = pred_resp.replace('null', 'None')
        if 'Answer:' in pred_resp:
            pred_resp = pred_resp.replace('Answer:', '')
        
        if self.config['framework'] == 'together':
            if '{' in pred_resp and '}' in pred_resp:
                start_index = pred_resp.index('{')
                end_index = pred_resp.index('}')
                pred_resp = pred_resp[start_index:end_index+1]

        if self.config['do_cot']:
            try:
                start_index = pred_resp.index('{')
                end_index = pred_resp.index('}')
                pred_resp = pred_resp[start_index:end_index+1]
            except ValueError:
                pred_resp = None

        try:
            pred_dict = ast.literal_eval(pred_resp)
        except:
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': [],
                'Sentence': ''
            }

        if not isinstance(pred_dict, dict):
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': [],
                'Sentence': ''
            }

        if 'Prediction' not in pred_dict.keys():
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': [],
                'Sentence': ''
            }
        if 'Intent' not in pred_dict.keys():
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': [],
                'Sentence': ''
            }
        
        if 'Sentence' not in pred_dict.keys():
            pred_dict = {
                'Prediction': '<SKIP>',
                'Intent': [],
                'Sentence': ''
            }

        yesno_pred = pred_dict["Prediction"]
        intent_pred = pred_dict["Intent"]
        

        binary_answer = self.map_binary_answer_to_int(yesno_pred)
        multiple_answer = self.map_multiple_answer_to_int(intent_pred)

        if self.yesno_to_int(output['label'].lower()) == binary_answer:
            result = True
        else:
            result = False

        copied_output = copy.deepcopy(output)
        copied_output['task1_binary_prediction'] = binary_answer
        copied_output['task1_binary_correct'] = self.yesno_to_int(output['label'].lower())
        copied_output['task1_binarized_model_answer'] = binary_answer
        copied_output['task1_binary_result'] = result


        photochat_plus_instance = self.photochat_plus[output['dialogue_id']]
        all_intent = photochat_plus_instance['all_info:intent']
        try:
            HA_intent = photochat_plus_instance['aggrement_info:intent']
        except KeyError:
            print(photochat_plus_instance.keys())
            assert False
        assert len(all_intent) > 0
        
        choice_answers = [self.intent_to_int(_intent.lower()) for _intent in all_intent]
        copied_output['task1_multiple_prediction'] = multiple_answer
        copied_output['task1_multiple_correct'] = choice_answers
        if len(multiple_answer) == 0:
            copied_output['task1_multiple_prediction_flag'] = False
        else:
            copied_output['task1_multiple_prediction_flag'] = True

        intent_f1, intent_precision, intent_recall = self.compute_intent_f1(multiple_answer, choice_answers)
        copied_output['task1_intent_f1'] = intent_f1
        copied_output['task1_intent_precision'] = intent_precision
        copied_output['task1_intent_recall'] = intent_recall


        if len(HA_intent) == 0:
            copied_output['task1_HA_intent_f1'] = 0
            copied_output['task1_HA_intent_precision'] = 0
            copied_output['task1_HA_intent_recall'] = 0
            copied_output['task1_HA_multiple_correct'] = []
            copied_output['task1_HA_multiple_flag'] = False
        else:
            HA_choice_answers = [self.intent_to_int(_intent.lower()) for _intent in HA_intent]
            HA_intent_f1, HA_intent_precision, HA_intent_recall = self.compute_intent_f1(multiple_answer, HA_choice_answers)
            copied_output['task1_HA_intent_f1'] = HA_intent_f1
            copied_output['task1_HA_intent_precision'] = HA_intent_precision
            copied_output['task1_HA_intent_recall'] = HA_intent_recall
            copied_output['task1_HA_multiple_correct'] = HA_choice_answers
            copied_output['task1_HA_multiple_flag'] = True

        pred_evidence = pred_dict['Sentence']

        gt_evidence = photochat_plus_instance['all_info:sentence_evidence']
        HA_gt_evidence = photochat_plus_instance['aggrement_info:sentence_evidence']

        f1_scores = []
        dist_scores = []
        for _gt in gt_evidence:
            f1_scores.append(self.compute_f1(pred_evidence, _gt))
            dist_scores.append(float(self.compute_dist(pred_evidence, _gt)))
        
        copied_output['task1_token_f1'] = sum(f1_scores)/len(f1_scores)
        copied_output['task1_dist'] = sum(dist_scores)/len(dist_scores)
        
        if len(HA_gt_evidence) == 0:
            copied_output['task1_HA_token_f1'] = 0.
            copied_output['task1_HA_dist'] = 0.
        else:
            HA_f1_scores = []
            HA_dist_scores = []
            for _gt in HA_gt_evidence:
                HA_f1_scores.append(self.compute_f1(pred_evidence, _gt))
                HA_dist_scores.append(float(self.compute_dist(pred_evidence, _gt)))
            
            copied_output['task1_HA_token_f1'] = sum(HA_f1_scores)/len(HA_f1_scores)
            copied_output['task1_HA_dist'] = sum(HA_dist_scores)/len(HA_dist_scores)

        return copied_output
    

    def evaluate_response(self, outputs):
        console.log("Running evaluation for task1...")

        final_outputs = []
        for output in tqdm(self.outputs, total=len(self.outputs)):
            
            if 'evidence' in self.config['template_name']:
                _output = self.evaluate_evidence_answer(output)
                if not _output:
                    continue
            elif 'multiple' in self.config['template_name']:
                _output = self.evaluate_multiple_choice_answer(output)
                if not _output:
                    continue
            else:
                _output = self.evaluate_binary_answer(output)
            final_outputs.append(_output)
        
        return final_outputs
    
    def score_and_analyze(self, df):
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        report = {}

        ############# Scores #############
        # Binary Answer
        binary_target_df = df[df['task1_binary_prediction'] != -1]
        model_responses = binary_target_df['task1_binary_prediction'].to_list()
        references = binary_target_df['task1_binary_correct'].to_list()

        report['binary-f1'] = f1_metric.compute(predictions=model_responses, references=references, pos_label=0, average="macro")['f1']
        report['binary-precision'] = precision_metric.compute(predictions=model_responses, references=references, pos_label=0, average="macro")['precision']
        report['binary-recall'] = recall_metric.compute(predictions=model_responses, references=references, pos_label=0, average="macro")['recall']

        if 'multiple' in self.config['template_name']:
            # Multiple Choice Answer
            # calculate the performance only for the "yes" case
            target_df = df[df['task1_binary_correct'] == 1]
            
            target_df = target_df[target_df['task1_multiple_prediction_flag'] == True]
            #report['choice-intersection'] = target_df['task1_multiple_choice_score'].mean()
            report['intent-f1'] = target_df['task1_intent_f1'].mean()
            report['intent-precision'] = target_df['task1_intent_precision'].mean()
            report['intent-recall'] = target_df['task1_intent_recall'].mean()

            report['HA_intent-f1'] = target_df['task1_HA_intent_f1'].mean()
            report['HA_intent-precision'] = target_df['task1_HA_intent_precision'].mean()
            report['HA_intent-recall'] = target_df['task1_HA_intent_recall'].mean()

            if 'evidence' in self.config['template_name']:
                # Evidence
                # calculate the performance only for the "yes" case
                report['token-f1'] = target_df['task1_token_f1'].mean()
                report['dist'] = target_df['task1_dist'].mean()

                report['HA_token-f1'] = target_df['task1_HA_token_f1'].mean()
                report['HA_dist'] = target_df['task1_HA_dist'].mean()

                total_len = len(df)
                all_target_df = df[
                    ((df['task1_binary_prediction'] == 1) & (df['task1_binary_correct'] == 1) &
                    (df['task1_multiple_prediction'] == df['task1_multiple_correct']) & 
                    (df['task1_token_f1'] > 0.5)) |
                    ((df['task1_binary_prediction'] == 0) & (df['task1_binary_correct'] == 0))
                ]
                all_target_len = len(all_target_df)

                report['all'] = all_target_len/total_len

                HA_df = df[df['task1_HA_multiple_flag'] == True]
                HA_total_len = len(HA_df)

                HA_all_target_df = df[
                    ((df['task1_binary_prediction'] == 1) & (df['task1_binary_correct'] == 1) &
                    (df['task1_multiple_prediction'] == df['task1_HA_multiple_correct']) & 
                    (df['task1_token_f1'] > 0.5)) |
                    ((df['task1_binary_prediction'] == 0) & (df['task1_binary_correct'] == 0))
                ]
                HA_all_target_len = len(HA_all_target_df)

                report['HA_all'] = HA_all_target_len/HA_total_len

        for history_len, sub_df in df.groupby("task1_history_len"):
            sub_pred = sub_df['task1_binary_prediction'].to_list()
            sub_refs = sub_df['task1_binary_correct'].to_list()

            sub_f1 = f1_metric.compute(predictions=sub_pred, references=sub_refs, pos_label=0, average="macro")['f1']

            report[f'{history_len}:binary-f1'] = sub_f1
        
        bins = [0, 5, 10, 15, 20, 25]
        group_names = ['0-5', '5-10', '10-15', '15-20', '20-25']
        df['History_Range'] = pd.cut(df['task1_history_len'], bins, labels=group_names)

        for history_len, sub_df in df.groupby("History_Range"):
            sub_pred = sub_df['task1_binary_prediction'].to_list()
            sub_refs = sub_df['task1_binary_correct'].to_list()

            sub_f1 = f1_metric.compute(predictions=sub_pred, references=sub_refs, pos_label=0, average="macro")['f1']

            report[f'{history_len}:binary-f1'] = sub_f1

        ## Error Analysis
        binary_wrong_reasons = df[(df['task1_binary_result'] == False)]['task1_binarized_model_answer'].value_counts(normalize=False).to_dict()

        if 0 in binary_wrong_reasons.keys():
            binary_wrong_reasons['false_negative'] = binary_wrong_reasons.pop(0)
        if 1 in binary_wrong_reasons.keys():
            binary_wrong_reasons['false_positive'] = binary_wrong_reasons.pop(1)
        if -1 in binary_wrong_reasons.keys():
            binary_wrong_reasons['irrelevant_response'] = binary_wrong_reasons.pop(-1)
        
        report['task1_binary_wrong_reasons_freq'] = binary_wrong_reasons
        
        for k, v in report.items():
            if isinstance(v, float):
                report[k] = round(v, 4) * 100
        
        return report

class Task2EvalAgent(BaseEvalAgent):

    def __init__(self, config: Dict) -> None:
        config['task_num'] = 'task2'

        super().__init__(config=config)

        self.descriptiveness_models = self.load_descriptiveness_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)


    def load_descriptiveness_model(self):
        return {
            'blipscore': RM.load_score("BLIP"), 
            'clipscore': RM.load_score("CLIP"),
            'image_reward': RM.load("ImageReward-v1.0")
        }

    def parse_objects(self, description: str):
        return description.split('Objects in the photo: ')[-1].split(',')

    def _run_object_detection(self, instance):
        pred_resp = instance['task2_generation']
        if "```python" in pred_resp:
            pred_resp = pred_resp.replace("```python", "").replace("```", "")
        try:
            pred_dict = ast.literal_eval(pred_resp)
        except:
            instance['task2_objects'] = None
            return instance

        llm_description = pred_dict['Image Description']
        output = call_chatgpt(OBJECT_TEMPLATE.format(llm_description=llm_description))
        
        instance['task2_objects'] = output.choices[0].message["content"]
        return instance
    

    def run_object_detection(self):
        outputs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for instance in tqdm(self.outputs, total=len(self.outputs)):
                copied_instance = copy.deepcopy(instance)
                
                future = executor.submit(self._run_object_detection, copied_instance)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                ret = future.result()
                outputs.append(ret)

        assert len(outputs) == len(self.outputs)
        return outputs

    
    def evaluate_completeness(self, golden_objects, pred_objects):
        if not pred_objects:
            return 0.

        pred_object_categories = list(pred_objects.keys())
        
        score = 0.
        for golden_object in golden_objects:
            if golden_object in pred_objects:
                score += 1.

        return score/len(golden_objects)
    
    def evaluate_descriptiveness(self, image_file_dir: str, llm_description: str):
        
        scores = {}
        for model_name, model_caller in self.descriptiveness_models.items():
            try:
                score = model_caller.score(llm_description.replace('An image of ', ''), image_file_dir)
                scores[model_name] = score
            except:
                scores[model_name] = 0

        return scores

    def evaluate_word_evidence(self, model_evidence, gt_evidence):
        f1_scores = []
        for _gt in gt_evidence:
            for _pred in model_evidence:
                f1_score = self.compute_f1(_pred, _gt)
                f1_scores.append(f1_score)

        return sum(f1_scores)/len(f1_scores)

    def evaluate_consistency(self, model_description, gt_descriptions):
        f1_scores = []
        dist_scores = []
        for gt_description in gt_descriptions:
            try:
                f1_score = self.compute_f1(model_description, gt_description)
                dist_scores.append(float(self.compute_dist(model_description, gt_description)))
            except:
                f1_score = 0.
                dist_scores.append(0.0)
            f1_scores.append(f1_score)
        
        return sum(f1_scores)/len(f1_scores), sum(dist_scores)/len(dist_scores)

    def dump_object_outputs(self, evaluation_outputs):
        """
        Dump the evaluation outputs
        """
        evaluated_response_filename = "object" + self.output_filename_suffix

        with open(os.path.join(self.output_dir, evaluated_response_filename), 'w') as f:
            json.dump(evaluation_outputs, f, ensure_ascii=False, indent='\t')

    def evaluate_response(self, outputs):
        console.log("Running evaluation for task2...")

        evaluated_response_filename = "object" + self.output_filename_suffix
        if os.path.isfile(os.path.join(self.output_dir, evaluated_response_filename)):
            with open(os.path.join(self.output_dir, evaluated_response_filename), 'r') as f:
                outputs = json.load(f)
        else:
            outputs = self.run_object_detection()
            self.dump_object_outputs(outputs)

        final_outputs = []
        for output in tqdm(outputs, total=len(outputs)):
            photo_description = output['photo_description']
            image_file_dir = output['image_file_dir']

            word_evidence = self.photochat_plus[output['dialogue_id']]['all_info:word_evidence']

            pred_resp = output['task2_generation']
            if "```python" in pred_resp:
                pred_resp = pred_resp.replace("```python", "").replace("```", "")
            try:
                pred_dict = ast.literal_eval(pred_resp)
                llm_description = pred_dict['Image Description']
            except:
                llm_description = None
            pred_objects = ['choices'][0]['message']['content']
            
            pred_objects = pred_objects.replace('null', 'None')
            
            try:
                pred_objects = ast.literal_eval(pred_objects)
            except SyntaxError:
                pred_objects = None
            except ValueError:
                pred_objects = None
            
            golden_objects = self.parse_objects(photo_description)

            complete_score = self.evaluate_completeness(golden_objects, pred_objects)

            descriptiveness_score = self.evaluate_descriptiveness(image_file_dir, llm_description)

            human_descriptions = self.photochat_plus[output['dialogue_id']]['all_info:user_description']

            consistency_score = self.evaluate_consistency(llm_description, human_descriptions)

            copied_output = copy.deepcopy(output)

            if 'evidence' in self.config['template_name']:
                pred_word_evidence = pred_dict['Words/Phrases']
                token_f1 = self.evaluate_word_evidence(pred_word_evidence, word_evidence)
                copied_output['task2_token_f1'] = token_f1
            
            copied_output['task2_complete_score'] = complete_score
            copied_output['task2_golden_objects'] = golden_objects
            copied_output['task2_llm_description'] = llm_description
            copied_output['task2_consistency_score:token_f1'] = consistency_score[0]
            copied_output['task2_consistency_score:dist'] = consistency_score[1]

            for k, v in descriptiveness_score.items():
                copied_output[f'task2_{k}'] = v

            final_outputs.append(copied_output)
        
        return final_outputs
    
    def score_and_analyze(self, df):
        report = {}

        ############# Scores #############
        # Descriptiveness
        blip_score = df['task2_blipscore']
        clip_score = df['task2_clipscore']
        image_reward = df['task2_image_reward']
        report['BLIPScore'] = blip_score.mean()
        report['CLIPScore'] = clip_score.mean()
        report['ImageReward'] = image_reward.mean()

        # Variability: Complete
        complete_score = df['task2_complete_score']
        report['Complete'] = complete_score.mean()
        
        # Consistency
        consistency_score_token_f1 = df['task2_consistency_score:token_f1']
        consistency_score_dist = df['task2_consistency_score:dist']
        report['Consistency:token-F1'] = consistency_score_token_f1.mean()
        report['Consistency:dist'] = consistency_score_dist.mean()

        # Word Evidence
        if 'evidence' in self.config['template_name']:
            report['word-evidence-token-f1'] = df['task2_token_f1'].mean()

        for k, v in report.items():
            if isinstance(v, float):
                if k == 'Complete':
                    report[k] = round(v, 4) * 100
                else:
                    report[k] = round(v, 4) 
        return report

class Task3EvalAgent(BaseEvalAgent):
    model_name_to_ckpt = {
        'clip-base-16': 'openai/clip-vit-base-patch16',
        'clip-base-32': 'openai/clip-vit-base-patch32',
        'clip-large-14': 'openai/clip-vit-large-patch14-336',
        'align-base': 'kakaobrain/align-base'
    }

    def __init__(self, config: Dict) -> None:
        
        super().__init__(config=config)

        self.model_ckpt = self.model_name_to_ckpt[config['t2i_model_name']]

        self.vllm_model_name = config['vllm_model_name']
        self.device = "cuda:{}".format(config['gpu_id'])
        self.model, self.processor, self.tokenizer = self.load_t2i_matching_model(self.model_ckpt)
        self.model.to(self.device)
        self.model.eval()
    
    def load_t2i_matching_model(self, model_ckpt):
        model = CLIPModel.from_pretrained(model_ckpt)
        processor = AutoProcessor.from_pretrained(model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        return model, processor, tokenizer

    def get_text_emb(self, text):
        inputs = self.tokenizer(
            text, 
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_emb = self.model.get_text_features(**inputs)
        return text_emb

    def get_image_emb(self, image_file):
        image = Image.open(image_file).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_emb = self.model.get_image_features(**inputs)
        return image_emb
    
    def evaluate_recall(self, logits, targets):
        batch_size, num_candidates = logits.shape
        sorted_indices = logits.sort(descending=True)[1]
        
        ranks = []
        for tgt, sort in zip(targets, sorted_indices.tolist()):
            rank = sort.index(tgt)
            ranks.append(rank)

        report = dict()
        for k in [1, 5, 10]:
            num_ok = 0
            for tgt, topk in zip(targets, sorted_indices[:, :k].tolist()):
                if tgt in topk:
                    num_ok += 1

            report[f'Recall@{k}'] = round((num_ok/batch_size), 4) * 100
        
        # MRR
        MRR = 0
        for tgt, topk in zip(targets, sorted_indices.tolist()):
            rank = topk.index(tgt) + 1
            MRR += 1/rank
        MRR = MRR/batch_size
        report['MRR'] = round(MRR, 4) * 100
        return report, ranks

    @torch.no_grad()
    def evaluate_response(self, outputs):
        preload_text_emb = torch.Tensor()
        preload_image_emb = torch.Tensor()
        preload_image_indices = []
        for i, output in enumerate(tqdm(outputs, total=len(outputs))):
            image_file_dir = output['image_file_dir']
            llm_description = output['task2_generation']
            
            if self.config['framework'] == 'together':
                print('wow', llm_description)
                llm_description = self.parse_llm_description(llm_description)
                assert llm_description != None, f'description: {llm_description}'
            
            text_feat = self.get_text_emb(llm_description)
            image_feat = self.get_image_emb(image_file_dir)

            # normalized features
            image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            
            preload_image_emb = torch.cat((preload_image_emb, image_feat.cpu()), 0)
            preload_text_emb = torch.cat((preload_text_emb, text_feat.cpu()), 0)

            preload_image_indices.append(i)

        logits = preload_text_emb @ preload_image_emb.t()
        recall_report, ranks = self.evaluate_recall(logits, preload_image_indices)
        
        new_outputs = []
        for output, rank in zip(outputs, ranks):
            copied_output = copy.deepcopy(output)
            copied_output['rank'] = rank
            new_outputs.append(copied_output)

        return new_outputs, recall_report

    def run(self):
        evaluated_outputs, reports = self.evaluate_response(self.outputs)
        self.dump_report_outputs(reports, evaluated_outputs)


class VLLMEvalAgent(Task3EvalAgent):
    def __init__(self, config):
        super().__init__(config)

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)

    def compute_vllm_dist(self, model_answer, ground_truth):
        pred_emb = self.embedder.encode(model_answer)
        gt_emb = self.embedder.encode(ground_truth)

        #similarity = cosine_similarity(pred_emb.reshape(1, -1), gt_emb.reshape(1, -1))[0][0]

        #return similarity
        return pred_emb, gt_emb

    def make_dialogue_history(self, output):
        dialogue = []
        for item in output['dialogue']:
            if item['share_photo']:
                break
            dialogue.append(item['message'])

        return '\n'.join(dialogue) #[-3:])

    @torch.no_grad()
    def evaluate_response(self, outputs):
        preload_text_emb = torch.Tensor()
        preload_image_emb = torch.Tensor()
        preload_image_indices = []
        for i, output in enumerate(tqdm(outputs, total=len(outputs))):
            image_file_dir = output['image_file_dir']
            
            dialogue_history = self.make_dialogue_history(output)
            vllm_save_dir = '<your save dir>'
            vllm_filename = image_file_dir.split('/')[-1].split('.jpg')[0]
            with open(os.path.join(vllm_save_dir, f'{self.vllm_model_name}_{vllm_filename}.txt'), 'r') as f:
                vllm_description = f.read()
            
            #vllm_text_feat = self.get_text_emb(vllm_description)
            vllm_text_feat = torch.Tensor(self.embedder.encode(vllm_description).reshape(1, -1))
            text_feat = torch.Tensor(self.embedder.encode(dialogue_history).reshape(1, -1))
            #text_feat = self.get_text_emb(dialogue_history)
            
            # normalized features
            #vllm_text_feat = vllm_text_feat / vllm_text_feat.norm(p=2, dim=-1, keepdim=True)
            #text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            
            preload_image_emb = torch.cat((preload_image_emb, vllm_text_feat.cpu()), 0)
            preload_text_emb = torch.cat((preload_text_emb, text_feat.cpu()), 0)

            preload_image_indices.append(i)

        logits = preload_text_emb @ preload_image_emb.t()
        recall_report = self.evaluate_recall(logits, preload_image_indices)
        return outputs, recall_report

    def run(self):
        evaluated_outputs, reports = self.evaluate_response(self.outputs)
        self.dump_report_outputs(reports, evaluated_outputs)

class CLIPEvalAgent(Task3EvalAgent):
    def __init__(self, config):
        super().__init__(config)

    def make_dialogue_history(self, output):
        dialogue = []
        for item in output['dialogue']:
            if item['share_photo']:
                break
            dialogue.append(item['message'])

        return '\n'.join(dialogue[-3:])

    @torch.no_grad()
    def evaluate_response(self, outputs):
        preload_text_emb = torch.Tensor()
        preload_image_emb = torch.Tensor()
        preload_image_indices = []
        for i, output in enumerate(tqdm(outputs, total=len(outputs))):
            image_file_dir = output['image_file_dir'].replace(
                '/yjlee/workspace/current_working/all_dataset_pool',
                '/work/workspace/LM_image_sharing/data/image_files'
            )
            
            dialogue_history = self.make_dialogue_history(output)
 
            text_feat = self.get_text_emb(dialogue_history)
            image_feat = self.get_image_emb(image_file_dir)

            # normalized features
            image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            
            preload_image_emb = torch.cat((preload_image_emb, image_feat.cpu()), 0)
            preload_text_emb = torch.cat((preload_text_emb, text_feat.cpu()), 0)

            preload_image_indices.append(i)

        logits = preload_text_emb @ preload_image_emb.t()
        recall_report = self.evaluate_recall(logits, preload_image_indices)
        return outputs, recall_report

    def run(self):
        evaluated_outputs, reports = self.evaluate_response(self.outputs)
        self.dump_report_outputs(reports, evaluated_outputs)

class HumanEvalAgent(Task3EvalAgent):
    def __init__(self, config):
        super().__init__(config)

        self.outputs = self.load_json('./photochat++/test.json')
        self.descriptiveness_models = self.load_descriptiveness_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)


    def load_descriptiveness_model(self):
        return {
            'blipscore': RM.load_score("BLIP"), 
            'clipscore': RM.load_score("CLIP"),
            'image_reward': RM.load("ImageReward-v1.0")
        }
    
    def parse_objects(self, description: str):
        return description.split('Objects in the photo: ')[-1].split(',')

    def _run_object_detection(self, instance):
        llm_description = instance['all_info:user_description'][0]
        output = call_chatgpt(OBJECT_TEMPLATE.format(llm_description=llm_description))
        
        instance['task2_objects'] = output
        return instance
    

    def run_object_detection(self):
        outputs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            
            for instance in tqdm(self.outputs, total=len(self.outputs)):
                copied_instance = copy.deepcopy(instance)
                
                future = executor.submit(self._run_object_detection, copied_instance)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                ret = future.result()
                outputs.append(ret)

        assert len(outputs) == len(self.outputs)
        return outputs

    def evaluate_completeness(self, golden_objects, pred_objects):
        if not pred_objects:
            return 0.

        pred_object_categories = list(pred_objects.keys())
        
        score = 0.
        for golden_object in golden_objects:
            if golden_object in pred_objects:
                score += 1.

        return score/len(golden_objects)
    
    def evaluate_descriptiveness(self, image_file_dir: str, llm_description: str):
        
        scores = {}
        for model_name, model_caller in self.descriptiveness_models.items():
            score = model_caller.score(llm_description.replace('An image of ', ''), image_file_dir)
            scores[model_name] = score

        return scores

    @torch.no_grad()
    def evaluate_response(self, outputs):
        evaluated_response_filename = "object" + self.output_filename_suffix
        if os.path.isfile(os.path.join(self.output_dir, evaluated_response_filename)):
            with open(os.path.join(self.output_dir, evaluated_response_filename), 'r') as f:
                outputs = json.load(f)
        else:
            outputs = self.run_object_detection()
            self.dump_object_outputs(outputs)

        final_outputs = []
        preload_text_emb = torch.Tensor()
        preload_image_emb = torch.Tensor()
        preload_image_indices = []
        for i, output in enumerate(tqdm(outputs, total=len(outputs))):
            all_description = output['all_info:user_description'][0]

            image_file_dir = output['image_file_dir']
            
            pred_objects = output['task2_objects']['choices'][0]['message']['content']
            
            pred_objects = pred_objects.replace('null', 'None')
            
            try:
                pred_objects = ast.literal_eval(pred_objects)
            except SyntaxError:
                pred_objects = None
            except ValueError:
                print(pred_objects)

            photo_description = output['photo_description']
            golden_objects = self.parse_objects(photo_description)

            complete_score = self.evaluate_completeness(golden_objects, pred_objects)

            descriptiveness_score = self.evaluate_descriptiveness(image_file_dir, all_description)
            copied_output = copy.deepcopy(output)
            copied_output['task2_complete_score'] = complete_score
            copied_output['task2_golden_objects'] = golden_objects
            
            for k, v in descriptiveness_score.items():
                copied_output[f'task2_{k}'] = v
            
            final_outputs.append(copied_output)
        
        return final_outputs

    def score_and_analyze(self, df):
        report = {}

        ############# Scores #############
        # Descriptiveness
        blip_score = df['task2_blipscore']
        clip_score = df['task2_clipscore']
        image_reward = df['task2_image_reward']
        report['BLIPScore'] = blip_score.mean()
        report['CLIPScore'] = clip_score.mean()
        report['ImageReward'] = image_reward.mean()

        # Variability: Complete
        complete_score = df['task2_complete_score']
        report['Complete'] = complete_score.mean()
        
        for k, v in report.items():
            if isinstance(v, float):
                if k == 'Complete':
                    report[k] = round(v, 4) * 100
                else:
                    report[k] = round(v, 4) 
        return report
    
    def dump_object_outputs(self, evaluation_outputs):
        """
        Dump the evaluation outputs
        """
        evaluated_response_filename = "object" + self.output_filename_suffix

        with open(os.path.join(self.output_dir, evaluated_response_filename), 'w') as f:
            json.dump(evaluation_outputs, f, ensure_ascii=False, indent='\t')
    
    def run(self):
        evaluated_outputs = self.evaluate_response(self.outputs)
        reports = self.run_reports(evaluated_outputs)
        self.dump_report_outputs(reports, evaluated_outputs)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    task_num = _config['task_num']
    if task_num == 'task1':
        evaluator = Task1EvalAgent(_config)
    elif task_num == 'task2':
        evaluator = Task2EvalAgent(_config)
    elif task_num == 'task3':
        evaluator = Task3EvalAgent(_config)
    elif task_num == 'vl':
        evaluator = CLIPEvalAgent(_config)
    elif task_num == 'vllm':
        evaluator = VLLMEvalAgent(_config)
    elif task_num == 'human':
        evaluator = HumanEvalAgent(_config)
    else:
        raise ValueError("Wrong task name!")

    evaluator.run()