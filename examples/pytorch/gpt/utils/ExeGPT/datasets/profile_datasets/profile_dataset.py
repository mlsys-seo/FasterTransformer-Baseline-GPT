import os, json, traceback
import multiprocessing

from pprint import pprint

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

NUM_PROC = int(multiprocessing.cpu_count() / 3)

DOWNLOAD_PATH = "/docker/data/datasets"
PROFILE_PATH = "./profile_results/len_datasets"
os.makedirs(PROFILE_PATH, exist_ok=True)


EVAL_DICT = {
    'mbpp': {
        "task": "CG",
        "load": "load_dataset('mbpp', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['text'])['input_ids'])",
        "output": "len(tokenizer(example['code'])['input_ids'])"
        },

    'wikisql': {
        "task": "CG",
        "load": "load_dataset('wikisql', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "len(tokenizer(example['sql']['human_readable'])['input_ids'])"
        },

    'openai_humaneval': {
        "task": "CG",
        "load": "load_dataset('openai_humaneval', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['prompt'])['input_ids'])",
        "output": "len(tokenizer(example['canonical_solution'])['input_ids'])"
        },

    'cnn_dailymail': {
        "task": "S",
        "load": "load_dataset('cnn_dailymail', '3.0.0', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['article'])['input_ids'])",
        "output": "len(tokenizer(example['highlights'])['input_ids'])"
        },
    
    # 'natural_questions': {
    #     "task": "C",
    #     "load": "load_dataset('natural_questions', 'dev', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['article'])['input_ids'])",
    #     "output": "len(tokenizer(example['highlights'])['input_ids'])"
    #     },

    # "databricks_dolly-qa": {
    #     "task": "C2",
    #     "load": "load_dataset('databricks/databricks-dolly-15k', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['instruction'])['input_ids']) + len(tokenizer(example['context'])['input_ids'])",
    #     "output": "len(tokenizer(example['response'])['input_ids'])",
    #     "filter": "lambda example: example['category'] in ['open_qa', 'closed_qa', 'brainstorming', 'general_qa']",
    #     },

    # "databricks_dolly-summarization": {
    #     "task": "S",
    #     "load": "load_dataset('databricks/databricks-dolly-15k', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['instruction'])['input_ids'])",
    #     "output": "len(tokenizer(example['response'])['input_ids'])",
    #     "filter": "lambda example: example['category'] in ['information_extraction', 'summarization']",
    #     },

    "truthful_qa-generation": {
        "task": "C2",
        "load": "load_dataset('truthful_qa', 'generation', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean([ len(tokenizer(seq)['input_ids']) for seq in example['correct_answers']+[example['best_answer']] ])",
        },

    "mmlu-anatomy": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'anatomy', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    "mmlu-astronomy": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'astronomy', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    "mmlu-college_biology": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'college_biology', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    "mmlu-college_computer_science": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'college_computer_science', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    "mmlu-college_mathematics": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'college_mathematics', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    # "mmlu-sociology": {
    #     "task": "C1",
    #     "load": "load_dataset('cais/mmlu', 'sociology', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['question'])['input_ids'])",
    #     "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
    #     },

    "mmlu-us_foreign_policy": {
        "task": "C1",
        "load": "load_dataset('cais/mmlu', 'us_foreign_policy', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['question'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
        },

    # "mmlu-professional_law": {
    #     "task": "C1",
    #     "load": "load_dataset('cais/mmlu', 'professional_law', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['question'])['input_ids'])",
    #     "output": "np.mean( [ len(tokenizer(output)['input_ids']) for output in example['choices'] ] )",
    #     },

    "hellaswag": {
        "task": "C2",
        "load": "load_dataset('hellaswag', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['ctx'])['input_ids'])",
        "output": "np.mean( [ len(tokenizer(sequence)['input_ids']) for sequence in example['endings'] ])",
        },

    # "ai2_arc": {
    #     "task": "C1",
    #     "load": "load_dataset('ai2_arc', 'ARC-Challenge', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['question'])['input_ids'])",
    #     "output": "np.mean( [ len(tokenizer(sequence)['input_ids']) for sequence in example['choices']['text'] ])",
    #     },

    # "openorca": {
    #     "task": "C",
    #     "load": "load_dataset('Open-Orca/OpenOrca', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['system_prompt']) + len(example['question'])['input_ids'])",
    #     "output": "np.mean( [ len(tokenizer(sequence)['input_ids']) for in sequence in example['response'] ]))",
    #     },

    "chatbot_arena_conversations": {
        "task": "C2",
        "load": "load_dataset('lmsys/chatbot_arena_conversations', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['conversation_a'][0]['content'])['input_ids'])",
        "output": "len(tokenizer(example['conversation_a'][1]['content'])['input_ids'])",
        },

    "big_patent": {
        "task": "S",
        "load": "load_dataset('big_patent', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['description'])['input_ids'])",
        "output": "len(tokenizer(example['abstract'])['input_ids'])",
        },
     
    "scientific_papers-arxiv": {
        "task": "S",
        "load": "load_dataset('./hf_datasets/scientific_papers.py', 'arxiv', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['article'])['input_ids'])",
        "output": "len(tokenizer(example['abstract'])['input_ids'])",
        },

    "scientific_papers-pubmed": {
        "task": "S",
        "load": "load_dataset('./hf_datasets/scientific_papers.py', 'pubmed', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['article'])['input_ids'])",
        "output": "len(tokenizer(example['abstract'])['input_ids'])",
        },
        
    'wmt16-cs-en': {
        "task": "T",
        "load": "load_dataset('wmt16', 'cs-en', 'validation', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['translation']['cs'])['input_ids'])",
        "output": "len(tokenizer(example['translation']['en'])['input_ids'])"
        },

    'wmt16-de-en': {
        "task": "T",
        "load": "load_dataset('wmt16', 'de-en', 'validation', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['translation']['de'])['input_ids'])",
        "output": "len(tokenizer(example['translation']['en'])['input_ids'])"
        },
    
    'wmt16-fi-en': {
        "task": "T",
        "load": "load_dataset('wmt16', 'fi-en', 'validation', cache_dir=DOWNLOAD_PATH)",
        "input": "len(tokenizer(example['translation']['fi'])['input_ids'])",
        "output": "len(tokenizer(example['translation']['en'])['input_ids'])"
        },
    
    # 'wmt16-ro-en': {
    #     "task": "T",
    #     "load": "load_dataset('wmt16', 'ro-en', 'validation', cache_dir=DOWNLOAD_PATH)",
    #     "input": "len(tokenizer(example['translation']['ro'])['input_ids'])",
    #     "output": "len(tokenizer(example['translation']['en'])['input_ids'])"
    #     },

    }

tokenizer = AutoTokenizer.from_pretrained(
    'facebook/opt-66b',
    cache_dir=DOWNLOAD_PATH,
    )

def get_len_eval_func(example, input_eval, output_eval):
    input_len = eval(input_eval)
    output_len = eval(output_eval)

    return {
        'input_len': input_len,
        'output_len': output_len
    }

def preprocess_seq_len(dataset, evals):
    if len(list(dataset.keys())):
        dataset = concatenate_datasets([dataset[s] for s in dataset.keys()])

    if "filter" in evals:
        dataset = dataset.filter(lambda example: eval(evals["filter"]))

    dataset = dataset.map(
            get_len_eval_func,
            num_proc=NUM_PROC,
            desc="get length",
            remove_columns=dataset.column_names,
            fn_kwargs={
                "input_eval": evals["input"],
                "output_eval": evals["output"],
            }
        )
    return dataset

if __name__=='__main__':
    dataset_names = list(EVAL_DICT.keys())
    failed_datasets = {}

    for dataset_name in dataset_names:
        try:
            FILE_PATH = f"{PROFILE_PATH}/{dataset_name}.json"

            if os.path.isfile(FILE_PATH):
                continue

            print("\n\n----------------------------------------------------")
            print(f"start {dataset_name}")
            dataset = eval(EVAL_DICT[dataset_name]["load"])
            dataset = preprocess_seq_len(dataset, EVAL_DICT[dataset_name])
            new_dataset_json = { key: dataset[key] for key in dataset.column_names }
            new_dataset_json['task'] = EVAL_DICT[dataset_name]['task']

            with open(FILE_PATH, 'w') as f:  
                json.dump(new_dataset_json, f, indent="\t")
                print(f"save {dataset_name}")

        except Exception as e:
            err_msg = traceback.format_exc()
            failed_datasets[dataset_name]: err_msg
            print(f"FAILED {dataset_name}")
            print(err_msg)

        if len(failed_datasets):
            print(f"FAILED:")
            for name, msg in failed_datasets.items():
                print(name)
                print(msg)
                print("==========\n\n")