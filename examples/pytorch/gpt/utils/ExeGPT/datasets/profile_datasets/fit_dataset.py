import os, json, traceback

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt

PROFILE_PATH = "./profile_results/len_datasets"


OUTLIER_THRESHOLD = 1.5
DIST_PATH = f"./profile_results/dists"
FIG_PATH = f'{DIST_PATH}/.cache'

os.makedirs(DIST_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

TARGET_TASK = {
    'S': {
        'input': 256,
        'output': 32,
        'max_input_ratio': 2,
        'max_output_ratio': 2.5,
    },

    'T': {
        'input': 128,
        'output': 128,
        'max_input_ratio': 2,
        'max_output_ratio': 2.5,
    },

    'C1': {
        'input': 256,
        'output': 64,
        'max_input_ratio': 2,
        'max_output_ratio': 2.5,
    },

    'C2': {
        'input': 512,
        'output': 256,
        'max_input_ratio': 2,
        'max_output_ratio': 2.5,
    },

    'CG': {
        'input': 64,
        'output': 192,
        'max_input_ratio': 2,
        'max_output_ratio': 2.5,
    },
}

SELECTED_DIST = {
    'ai2_arc':{
        'input': 'truncnorm',
        'output': 'truncnorm',
    },
    'big_patent':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'wikisql':{
        'input': 'norm',
        'output': 'norm',
    },
    'chatbot_arena_conversations':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'cnn_dailymail':{
        'input': 'truncnorm',
        'output': 'norm',
    },
    'databricks_dolly-qa':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'databricks_dolly-summarization':{
        'input': 'truncnorm',
        'output': 'truncnorm',
    },
    'hellaswag':{
        'input': 'norm',
        'output': 'norm',
    },
    'mbpp':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'mmlu-anatomy':{
        'input': 'norm',
        'output': 'norm',
    },
    'mmlu-astronomy':{
        'input': 'norm',
        'output': 'norm',
    },
    'mmlu-college_biology':{
        'input': 'norm',
        'output': 'norm',
    },
    'mmlu-college_computer_science':{
        'input': 'norm',
        'output': 'norm',
    },
    'mmlu-college_mathematics':{
        'input': 'norm',
        'output': 'norm',
    },
    'mmlu-professional_law':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'mmlu-sociology':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'mmlu-us_foreign_policy':{
        'input': 'norm',
        'output': 'norm',
    },
    'openai_humaneval':{
        'input': 'norm',
        'output': 'norm',
    },
    'scientific_papers-arxiv':{
        'input': 'norm',
        'output': 'norm',
    },
    'scientific_papers-pubmed':{
        'input': 'truncnorm',
        'output': 'norm',
    },
    'truthful_qa-generation':{
        'input': 'norm',
        'output': 'norm',
    },
    'wmt16-cs-en':{
        'input': 'norm',
        'output': 'truncnorm',
    },
    'wmt16-de-en':{
        'input': 'truncnorm',
        'output': 'norm',
    },
    'wmt16-ro-en':{
        'input': 'truncnorm',
        'output': 'truncnorm',
    },
    'wmt16-fi-en':{
        'input': 'truncnorm',
        'output': 'truncnorm',
    },
}

def fit_dists(data_np):
    # normal
    # mean, std
    norm_args = norm.fit(data_np)
    norm_dist = norm(*norm_args)
    norm_rmse = np.sqrt(np.mean((norm_dist.pdf(data_np) - np.ones_like(data_np)) ** 2))

    # truncated normal
    # fit_a, fit_b, truncated_mean, truncated_std

    # def trunc_func(p, r, xa, xb):
    #     return truncnorm.nnlf(p, r)

    # def constraint(p, r, xa, xb):
    #     a, b, loc, scale = p
    #     return np.array([a*scale + loc - xa, b*scale + loc - xb])
    
    # xa, xb = float(min(data_np)), float(max(data_np))
    # print(f"xa:{xa}, xb:{xb}")
    # loc_guess = norm_args[0]
    # scale_guess = norm_args[1]
    # a_guess = (xa - loc_guess)/scale_guess
    # b_guess = (xb - loc_guess)/scale_guess
    # p0 = [a_guess, b_guess, loc_guess, scale_guess]

    # truncnorm_args = fmin_slsqp(trunc_func, p0, f_eqcons=constraint, args=(data_np, xa, xb),
    #              iprint=False, iter=1000)
    # print(p0)
    # print(truncnorm_args)
    # input()


    xa, xb = float(min(data_np)), float(max(data_np))
    loc_guess = norm_args[0]
    scale_guess = norm_args[1]
    a_guess = (xa - loc_guess)/scale_guess
    b_guess = (xb - loc_guess)/scale_guess

    # truncnorm_args = truncnorm.fit(data_np)
    truncnorm_args = truncnorm.fit(data_np, scale= scale_guess, loc = loc_guess)
    truncnorm_dist = truncnorm(*truncnorm_args)
    truncnorm_rmse = np.sqrt(np.mean((truncnorm_dist.pdf(data_np) - np.ones_like(data_np)) ** 2))

    # print(f"xa:{xa}, xb:{xb}, a_guess:{a_guess}, b_guess:{b_guess}, scale_guess:{scale_guess}, loc_guess:{loc_guess}")
    # print(f"truncnorm_args: {truncnorm_args}")
    # input()
    dist_args = {
        'norm': norm_args,
        'truncnorm': truncnorm_args,
    }

    val_99 = {
        'norm': norm_dist.ppf(0.99),
        'truncnorm': truncnorm_dist.ppf(0.99),
    }

    scores = {
        'norm': norm_rmse,
        'truncnorm': truncnorm_rmse,
    }

    if truncnorm_args[2] < 0:
        best_dist_name = 'truncnorm'
    else:
        best_dist_name = 'norm'

    return dist_args, best_dist_name


def scale_dist_to_tasks(dist_name, dist_args, target_mean):
    return None


def get_99p_length(dist_name, dist_args):
    if dist_name == 'norm':
        percentile_value = norm.ppf(0.99, *dist_args)
    return percentile_value


def get_IQR_outliers_idx(data, threshold=1.5):
    quartile_1 = np.percentile(data, 25)
    quartile_3 = np.percentile(data, 75)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (threshold * iqr)
    upper_bound = quartile_3 + (threshold * iqr)
    return np.asarray((data < lower_bound) | (data > upper_bound)).nonzero()[0]


def get_nx_mean_outliers_idx(data, threshold=1.5):
    upper_bound = np.mean(data) * threshold
    return np.asarray(data>upper_bound).nonzero()[0]

def remove_outliers(input_len, output_len, threshold=1.5):
    input_outliers_idx = get_IQR_outliers_idx(input_len, threshold)
    output_outliers_idx = get_IQR_outliers_idx(output_len, threshold)
    
    # idx_to_remove = np.unique(np.concatenate(input_outliers_idx, output_outliers_idx))
    input_len = np.delete(input_len, input_outliers_idx)
    output_len = np.delete(output_len, output_outliers_idx)
    return input_len, output_len


def draw_plot(name, data, dist_args):
    plt.figure(figsize=(10, 6))
    num_bins = int(min(100, max(data)-min(data)))
    x = np.linspace(np.min(data), np.max(data), num_bins)

    plt.hist(data, bins=num_bins, alpha=0.5, label=f'{name}', color='gray', density=True)

    if dist_args is not None:
        plt.plot(x, truncnorm.pdf(x, *dist_args['truncnorm']), 'r-', lw=3, alpha=0.5, label='Fitted Trunc-Normal Distribution')
        plt.plot(x, norm.pdf(x, *dist_args['norm']), 'b--', lw=1, alpha=1.0, label='Fitted Normal Distribution')

    plt.xlabel('seq_len')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {name}')
    plt.legend()

    
    plt.savefig(f'{FIG_PATH}/hist_{name}.png')


if __name__=='__main__':
    
    dataset_filenames = [ filename for filename in os.listdir(PROFILE_PATH) if ".json" in filename ]
    

    dist_results = []
    failed_datasets = {}
    for filename in dataset_filenames:
        try:
            print(f"--------------------------fitting data from {filename}--------------------------")
            dataset_name = filename.split(".json")[-2]
            FILE_PATH = f"{PROFILE_PATH}/{filename}"

            if not os.path.isfile(FILE_PATH):
                continue

            with open(FILE_PATH, 'r') as f:  
                data_len_dict = json.load(f)

            input_len = np.array(data_len_dict['input_len'])
            output_len = np.array(data_len_dict['output_len'])
            task_name = data_len_dict['task']

            input_len = input_len[~np.isnan(input_len)]
            output_len = output_len[~np.isnan(output_len)]

            corrcoef = np.corrcoef(input_len, output_len)[0][1]

            print(f"before cleaning: count {len(input_len)} input_mean {round(np.mean(input_len),2)}, input_max {np.max(input_len)} / output_mean {round(np.mean(output_len))}, output_max {np.max(output_len)}")
            input_len, output_len = remove_outliers(input_len, output_len, OUTLIER_THRESHOLD)
            print(f"after cleaning: count {len(input_len)} input_mean {round(np.mean(input_len),2)}, input_max {np.max(input_len)} / output_mean {round(np.mean(output_len))}, output_max {np.max(output_len)}")

            input_dist_args, input_dist_name, = fit_dists(input_len)
            output_dist_args, output_dist_name, = fit_dists(output_len)
            input_dist_name = SELECTED_DIST[dataset_name]['input']
            output_dist_name = SELECTED_DIST[dataset_name]['output']

            input_dist_mean_std = input_dist_args[input_dist_name] if input_dist_name == "norm" else input_dist_args[input_dist_name][2:]
            output_dist_mean_std = output_dist_args[output_dist_name] if output_dist_name == "norm" else output_dist_args[output_dist_name][2:]

            adj_input_ratio = TARGET_TASK[task_name]['input'] / input_dist_mean_std[0]
            adj_output_ratio = TARGET_TASK[task_name]['output'] / output_dist_mean_std[0]

            input_adj_mean = input_dist_mean_std[0] * adj_input_ratio
            input_adj_max = max(input_len) * abs(adj_input_ratio)
            input_adj_std = input_dist_mean_std[1] * abs(adj_input_ratio)
            input_adj_99 = input_adj_std*2.33 + input_adj_mean

            output_adj_mean = output_dist_mean_std[0] * adj_output_ratio
            output_adj_max = max(output_len) * abs(adj_output_ratio)
            output_adj_std = output_dist_mean_std[1] * abs(adj_output_ratio)
            output_adj_99 = output_adj_std*2.33 + output_adj_mean

            dist_results.append({
                'task': task_name,
                'dataset_name': dataset_name,
                'corr': corrcoef,
                'out/in_ratio': np.mean(output_len)/np.mean(input_len),

                'input_mean': np.mean(input_len),
                'input_std': np.std(input_len),
                'input_max': max(input_len),
                'input_max/mean': max(input_len)/np.mean(input_len),
                'input_dist': input_dist_name,
                'input_dist_mean': input_dist_mean_std[0],
                'input_dist_std': input_dist_mean_std[1],
                'input_args': input_dist_args[input_dist_name],

                'output_mean': np.mean(output_len),
                'output_std': np.std(output_len),
                'output_max': max(output_len),
                'output_max/mean': max(output_len)/np.mean(output_len),
                'output_dist': output_dist_name,
                'output_dist_mean': output_dist_mean_std[0],
                'output_dist_std': output_dist_mean_std[1],
                'output_args': output_dist_args[output_dist_name],

                'adj_input_ratio': adj_input_ratio,
                'input_adj_mean': input_adj_mean,
                'input_adj_true_min': min(input_len) * adj_input_ratio,
                'input_adj_true_max': input_adj_max,
                'input_adj_max': TARGET_TASK[task_name]['max_input_ratio'] * input_adj_mean,
                'input_adj_std': input_adj_std,
                'input_adj_99': input_adj_99,

                'adj_output_ratio': adj_output_ratio,
                'output_adj_mean': output_adj_mean,
                'output_adj_true_min': min(output_len) * adj_output_ratio,
                'output_adj_true_max': output_adj_max,
                'output_adj_max': TARGET_TASK[task_name]['max_output_ratio'] * output_adj_mean,
                'output_adj_std': output_adj_std,
                'output_adj_99': output_adj_99,
                
            })

            draw_plot(f'{dataset_name}_input', input_len, input_dist_args)
            draw_plot(f'{dataset_name}_output', output_len, output_dist_args)

        except Exception as e:
                err_msg = traceback.format_exc()
                failed_datasets[dataset_name]: err_msg
                print(f"FAILED {dataset_name}")
                print(err_msg)
    
    result_pd = pd.DataFrame(dist_results)
    result_pd = result_pd.round(2)
    result_pd.to_csv(f"{DIST_PATH}/fit_result.csv")
    print(f"saved {DIST_PATH}/fit_result.csv")

    if len(failed_datasets):
        print(f"FAILED:")
        for name, msg in failed_datasets.items():
            print(name)
            print(msg)
            print("==========\n\n")