import os, json
import math, random

from scipy.linalg import cholesky
from scipy.stats import norm, truncnorm
from scipy.optimize import minimize

import torch
import numpy as np

from numpy.random import default_rng

TASK_CONFIG_PATH = "/workspace/down/BBB/BBB_FasterTransformer/examples/pytorch/gpt/utils/ExeGPT/datasets/config_task/tasks.json"
TASK_CONFIG= None
RANDOM_RNG=default_rng(12341234)
np.random.seed(12341234)

if os.path.isfile(TASK_CONFIG_PATH):
    with open(TASK_CONFIG_PATH, 'r') as f:  
        TASK_CONFIG = json.load(f)



class ExeGPTSimulatedDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, return_only_len=False):
        assert task_name in TASK_CONFIG, f"task_name is not in out task: {TASK_CONFIG.keys()}"
        task_config = TASK_CONFIG[task_name]

        self.task_name = task_name
        self.return_only_len = return_only_len

        self.dist = task_config["dist"] if "dist" in task_config else "truncnorm"
        self.coef = task_config["coef"]

        self.in_desired_params = self.get_desired_params(task_config["input"])
        self.out_desired_params = self.get_desired_params(task_config["output"])

        self.in_seq_min, self.in_seq_max, self.in_seq_mean, self.in_seq_std = self.in_desired_params
        self.out_seq_min, self.out_seq_max, self.out_seq_mean, self.out_seq_std = self.out_desired_params

        if self.dist == "norm":
            self.in_params = self.in_desired_params[2:]
            self.out_params = self.out_desired_params[2:]
            self.in_dist, self.out_dist = self.init_norm(self.in_desired_params, self.out_desired_params)
        else:
            self.in_params = task_config["input"]["dist_params"] if "dist_params" in task_config["input"] else self.truncated_normal_params(*self.in_desired_params)
            self.out_params = task_config["output"]["dist_params"] if "dist_params" in task_config["output"] else self.truncated_normal_params(*self.out_desired_params)
            self.in_dist, self.out_dist = self.init_truncnorm(self.in_params, self.out_params)
        
        self.in_seq_99 = round(self.in_dist.ppf(0.99))
        self.out_seq_99 = round(self.out_dist.ppf(0.99))

        # print("\nINPUT DIST")
        # print(":: desired attributes")
        # print(f":: {self.in_desired_params}")
        # print(":: params")
        # print(f":: {self.in_params}")

        # print("\nOUTPUT DIST")
        # print(":: desired attributes")
        # print(f":: {self.out_desired_params}")
        # print(":: params")
        # print(f":: {self.out_params}")
        # print('\n------------------ ExeGPTDataset ------------------')
        # print(f":: {'task'.ljust(25, ' ')}: {self.task_name}")
        # print(f":: {'dist'.ljust(25, ' ')}: {self.dist}")
        # print(f":: {'mean'.ljust(25, ' ')}: in {self.in_seq_mean} | out {self.out_seq_mean}")
        # print(f":: {'std'.ljust(25, ' ')}: in {self.in_seq_std} | out {self.out_seq_std}")
        # print(f":: {'99% of seq_len'.ljust(25, ' ')}: in {self.in_seq_99} | out {self.out_seq_99}")
        # print('---------------------------------------------------\n')

    
    def get_desired_params(self, current_config):
        seq_mean = current_config['seq_mean']
        seq_std = current_config['seq_std']
        seq_min = current_config['seq_min'] - 0.499
        seq_max = int(seq_mean * current_config['max_ratio']) + 0.5
        return seq_min, seq_max, seq_mean, seq_std


    def init_norm(self, in_params, out_params):
        in_dist = norm(*in_params)
        out_dist = norm(*out_params)
        return in_dist, out_dist


    def init_truncnorm(self, in_params, out_params):
        in_dist = truncnorm(*in_params)
        out_dist = truncnorm(*out_params)
        return in_dist, out_dist


    def truncated_normal_params(self, truncation_min, truncation_max, desired_mean, desired_std):
        print("---")
        def error_function(params):
            mu, sigma = params
            dist = truncnorm((truncation_min - mu) / sigma, 
                                (truncation_max - mu) / sigma, 
                                loc=mu, 
                                scale=sigma)
            mean, variance = dist.stats(moments='mv')
            return (mean - desired_mean)**2 + (variance - desired_std**2)**2

        # Initial guess (normal distribution parameters)
        initial_guess = [desired_mean, desired_std]

        # Run the optimizer
        solution = minimize(error_function, initial_guess, method='nelder-mead')

        mu, sigma = solution.x

        a = (truncation_min - mu) / sigma 
        b = (truncation_max - mu) / sigma

        return a, b, mu, sigma


    def __len__(self):
        return np.inf


    def __getitem__(self, idx):
        in_len, out_len = self.get_in_out_lengths(1)
        return in_len[0], out_len[0]
    

    def in_ppf(self, *args):
        return self.in_dist.ppf(*args)    


    def out_ppf(self, *args):
        return self.out_dist.ppf(*args)    

    
    def get_input_prob(self, x:int):
        return self.in_dist.cdf(x+0.5) - self.in_dist.cdf(x-0.5)


    def get_output_prob(self, x:int):
        return self.out_dist.cdf(x+0.5) - self.out_dist.cdf(x-0.5)


    def get_sample(self):
        in_len, out_len = self.__getitem__(1)

        if self.return_only_len:
            return in_len, out_len
        
        in_seq_list = [ random.randrange(5, 100) for _ in range(in_len) ]
        in_seq = torch.IntTensor(in_seq_list)
        out_seq_len = torch.IntTensor([out_len])

        return in_seq, out_seq_len


    def get_batch(self, batch_size):

        in_lens, out_lens = self.get_in_out_lengths(batch_size)
        
        if self.return_only_len:
            return in_lens, max(out_lens)

        max_len = max(in_lens)
        in_seq_list = [ torch.ones(max_len) * 999 for _ in in_lens ]
        for idx, in_len in enumerate(in_lens):
            in_seq_list[idx][:in_len] = np.random.randint(3,10)
        out_seq_len = torch.IntTensor(out_lens)
        
        # in_seq_list = [torch.IntTensor([10 for _ in range(20)]) for _ in range(batch_size)]
        # out_seq_len = torch.IntTensor([42 for _ in range(batch_size)])

        return in_seq_list, max(out_lens)
    

    def generate_truncated_normal_samples(self, mean, std, lower_bound, upper_bound, size):
        a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
        samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        return samples


    def get_in_out_lengths(self, batch_size):


        """Apply Coef: Trial Method 1
        in_lens = truncnorm.rvs(*self.in_params, size=batch_size)
        # Calculate the mean and standard deviation of the second truncated normal distribution
        mean_adj = self.out_seq_mean + self.coef * self.out_seq_std * (self.in_seq_mean - self.in_seq_min) / self.in_seq_std
        std_adj = np.sqrt((self.out_seq_std ** 2) * (1 - self.coef ** 2))

        # Generate samples from the second truncated normal distribution
        out_lens = self.generate_truncated_normal_samples(mean_adj, std_adj, self.out_seq_min, self.out_seq_max, batch_size)
        """

        """Apply Coef: Trial Method 2
        # independent_rvs = np.random.normal(0, 1, size=(2, batch_size))
        
        # # Apply Correlation between in_seq_len and out_seq_len
        # corr_matrix = np.array([
        #     [1, self.coef],
        #     [self.coef, 1]
        # ])
        # lower_triangular = cholesky(corr_matrix, lower=True)
        # correlated_std_normals = np.dot(lower_triangular, independent_rvs)

        # in_lens = self.in_dist.ppf(self.in_dist.cdf(correlated_std_normals[0]))
        # out_lens = self.out_dist.ppf(self.out_dist.cdf(correlated_std_normals[1]))
        """

        in_lens = self.in_dist.rvs(size=batch_size, random_state=RANDOM_RNG)
        out_lens = self.out_dist.rvs(size=batch_size, random_state=RANDOM_RNG)

        in_lens = list(map(round, in_lens))
        out_lens = list(map(round, out_lens))

        return in_lens, out_lens


# dataset = ExeGPTDataset()
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


if __name__ == "__main__":
    task_names = list(TASK_CONFIG.keys())
    print(task_names)


    task_name = "C1"

    """
    return_only_len=True: return seq_len only
    """
    task_dataset = ExeGPTSimulatedDataset(task_name, return_only_len=True)

    input_len, output_len = task_dataset.get_sample()
    # input_len: 247 output_len: 103

    input_lens, output_lens = task_dataset.get_batch(4)
    # input_lens: [317, 405, 142, 115] output_lens: [111, 58, 41, 134]
    

    """
    return_only_len=False: return torch.Tensor with tokens and padding
    """
    task_dataset = ExeGPTSimulatedDataset(task_name, return_only_len=False)

    input_len, output_len = task_dataset.get_sample()
    # input_len: tensor([65, 44, 91, 73], dtype=torch.int32)
    # output_len: tensor([4], dtype=torch.int32)
    
    input_lens, output_lens = task_dataset.get_batch(3)
    # input_lens: tensor([[65, 44, 91, 73], [65, 44, 91, 0], [65, 44, 0, 0]], dtype=torch.int32)
    # output_lens: tensor([4, 3, 2], dtype=torch.int32)
    
    
    
    for task_name in task_names:
        print(f"\n\n----------------------------------")
        print(f"Task name: {task_name}")
        task_dataset = ExeGPTSimulatedDataset(task_name, return_only_len=True)

        input_lens, output_lens = task_dataset.get_batch(10)
        print(f" INPUT:: mean={round(np.mean(input_lens), 2)}, std={round(np.std(input_lens), 2)}")
        print(f"OUTPUT:: mean={round(np.mean(output_lens), 2)}, std={round(np.std(output_lens), 2)}")