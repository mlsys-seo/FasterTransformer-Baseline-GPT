import argparse
import multiprocessing
from itertools import repeat
import itertools

from tqdm import tqdm
import parmap
import pandas as pd

from profiler import initialize_profiler, run_profile_bsize, query_latency, get_TPType, estimate_decoder_batch


MODEL_LAYER_NUM = {
    'T5': {
        '11B': 48,
        '22B': 96,
    },
    'GPT': {
        '13B': 40,
        '39B': 48,
        '101B': 80,
        '175B': 96,
        '341B': 120,
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='GPT', choices=['GPT', 'T5'], help='')
    parser.add_argument('--model_size', type=str, default='39B', help='')
    parser.add_argument('--gpu_type', type=str, default='A40', help='')
    parser.add_argument('--input_seq_len', type=int, default=128, help='')
    parser.add_argument('--output_seq_len', type=int, default=64, help='')
    parser.add_argument('--num_gpus', type=int, default=8, help='')
    # parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    args = parser.parse_args()

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    return args

def get_latency(num_sublayer, batch_size, is_encoder=False, num_tp=1):
    return query_latency(num_sublayer, int(batch_size), is_encoder, get_TPType(num_tp))


def rolling_max(array, n=4):
    len_array = len(array)
    array = array[-n+1:] + array
    out_array = [ max(array[idx:idx+n]) for idx in range(len_array) ]
    return out_array

def get_latency_for_window_size(window_size, num_tp, num_pp, num_sublayer):
    num_gpus = num_tp * num_pp
    result_list = []
    for encoder_bsize in range(1, 250):
        # bsizes_in_window = run_profile_bsize(encoder_bsize, args.output_seq_len, window_size)
        bsizes_in_window, _ = estimate_decoder_batch(args.output_seq_len, encoder_bsize, args.output_seq_len*2, window_size)

        encoder_latency = get_latency(num_sublayer=num_sublayer, batch_size=encoder_bsize, is_encoder=True, num_tp=num_tp)
        decoder_latencies = [
                get_latency(num_sublayer=num_sublayer, batch_size=bsize, is_encoder=False, num_tp=num_tp) for bsize in bsizes_in_window
            ]

        # print(f"encoder latency: {encoder_latency}")
        # print(f"decoder_latencies: {decoder_latencies}")

        stage_latencies = [encoder_latency] + decoder_latencies # encoder_time + (window_size * decoder_time)

        if None in stage_latencies:
            result_list.append(None)
            continue
        
        stage_latencies = rolling_max(stage_latencies, num_gpus) 
        stage_latency = sum(stage_latencies)
        latency = stage_latency * (args.output_seq_len // window_size) + sum(stage_latencies[:args.output_seq_len % window_size])
        
        # print(bsizes_in_window)
        result_list.append({
            'tp': num_tp,
            'pp': num_pp,
            'encoder_bsize': encoder_bsize,
            'window_size': window_size,
            'max_bsizes_in_window': max(bsizes_in_window),
            'stage_latency': stage_latency/1000,
            'latency': round(latency, 2)/1000,
            'token_latency': round((latency/1000) / args.output_seq_len, 2),
            'throughput': round(encoder_bsize / (stage_latency/1000), 2),
        })
    return result_list

def run_latency_profile(num_tp, num_pp, num_sublayer):

    window_sizes = list(range(1, args.output_seq_len))
    NUM_CORES = int(multiprocessing.cpu_count() / 4)

    result_list = parmap.map(	
        get_latency_for_window_size,
        window_sizes,
        num_tp, num_pp, num_sublayer,
        pm_pbar=True,
        pm_processes=NUM_CORES)

    # with Pool(num_pool) as p:
    #     result_list = p.starmap(get_latency_for_encoder_bsize, zip(encoder_bsizes, repeat(num_tp), repeat(num_pp), repeat(num_sublayer) ))
    result_list = list(itertools.chain.from_iterable(result_list))
    return list(filter(lambda item: item is not None, result_list))

if __name__ == '__main__':
    args = parse_args()


    # encoder_bsize = 32
    # args.output_seq_len = 1024 # 비례
    # window_size = 1 # 반비례

    # 128 - 4
    # 64 - 2

    # jkim = run_profile_bsize(encoder_bsize, args.output_seq_len, window_size)
    # kkh, _ = estimate_decoder_batch(args.output_seq_len, encoder_bsize, args.output_seq_len*2, window_size)

    # diff = [ round((jkim[idx] - kkh[idx]) / jkim[idx], 2) for idx in range(len(jkim)) ]
    

    # print(jkim)
    # print(kkh)
    # print(diff)
    # print(round(sum(diff)/len(diff), 2))
    # print("-----------------------\n")
    
    profile_dir_path = f"./profile-results/{args.model_type}/{args.model_size}/{args.input_seq_len}-{args.output_seq_len}/{args.gpu_type}"
    initialize_profiler(profile_dir_path, tp=None)

    result_list = []
    for num_tp in [8]:
        num_pp = int(args.num_gpus // num_tp)

        num_sublayer = MODEL_LAYER_NUM[args.model_type][args.model_size] / args.num_gpus
        assert num_sublayer % 1 == 0, "a number of model is not divided by a number of gpus" 

        result_list += run_latency_profile(num_tp=num_tp, num_pp=num_pp, num_sublayer=num_sublayer)
    

    result_pd = pd.DataFrame(result_list)
    result_pd.to_csv(f"./{args.model_type}{args.model_size}_{args.input_seq_len}-{args.output_seq_len}_{args.gpu_type}.csv")
    