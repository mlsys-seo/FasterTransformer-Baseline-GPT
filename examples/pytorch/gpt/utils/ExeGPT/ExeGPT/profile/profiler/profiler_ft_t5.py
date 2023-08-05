import subprocess
import argparse
import os
import re
import json


def launch(path, save_path, file_name, cmd):
    p = subprocess.run(cmd, cwd=path, shell=True, stdout=subprocess.PIPE)
    print(p.stdout.decode('UTF-8'))
    profiled_data = get_data(p.stdout.decode('UTF-8'))
    with open(os.path.join(save_path, f"{file_name}.json"), 'w') as f:
        json.dump(profiled_data, f, indent=4)
        
def get_data(output):
    encoder_pattern = r"(\d+)(\-Encoder\-)(\d+\.\d+)"
    encoder_repatter = re.compile(encoder_pattern)
    encoder_match = encoder_repatter.findall(output)
    
    kv_pattern = r"(\d+)(\-Decoder\-With\_KV\-)(\d+\.\d+)"
    kv_repatter = re.compile(kv_pattern)
    kv_match = kv_repatter.findall(output)
    
    decoder_pattern = r"(\d+)(\-Decoder\-Without\_KV\-)(\d+\.\d+)"
    decoder_repatter = re.compile(decoder_pattern)
    decoder_match = decoder_repatter.findall(output)
    
    profiled_data = {}
    
    for encoder_match, kv_match, decoder_match in zip(encoder_match, kv_match, decoder_match):
        batch = int(encoder_match[0])
        encoder_time = float(encoder_match[-1])
        decoder_time = float(decoder_match[-1])
        kv_time = round(float(kv_match[-1]) - decoder_time, 5)
        
        profiled_data[batch] = {"Encoder": encoder_time,
                                "Decoder": decoder_time,
                                "KV": kv_time}
    
    profiled_data = dict(sorted(profiled_data.items()))
    
    return profiled_data
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    parser.add_argument("--encoder-seq-length", type=int)
    parser.add_argument("--decoder-seq-length", type=int)

    parser.add_argument("--start-batch-size", type=int)
    parser.add_argument("--end-batch-size", type=int)
    parser.add_argument("--batch-size-hop", type=int)

    parser.add_argument("--sample-iters", type=int)
    parser.add_argument("--save-file-path", type=str)
    parser.add_argument("--ft-path", type=str)
    parser.add_argument("--config-path", type=str)
    
    parser.add_argument("--num_layers", type=int)

    parser.add_argument('--fp16', action='store_true',
                        default=False)

    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()
    argument_cmd = f"{os.path.join(args.ft_path, 'examples/pytorch/t5/mnli_task_example.py')} \
                --start_batch_size {args.start_batch_size} \
                --end_batch_size {args.end_batch_size} \
                --batch_size_hop {args.batch_size_hop} \
                --encoder_max_seq_len {args.encoder_seq_length} \
                --decoder_max_seq_len {args.decoder_seq_length} \
                --data_type {'fp16' if args.fp16 else 'fp32'} \
                --model {args.model} \
                --profile_iters {args.sample_iters} \
                --config_path {args.config_path}"

    #Single GPU
    launch_cmd = f"python " + argument_cmd
    launch(args.ft_path, args.save_file_path, "single_result", launch_cmd)
    
    #TP
    launch_cmd = f"mpirun -n 2 --allow-run-as-root python " + argument_cmd + " --tensor_para_size 2 --pipeline_para_size 1"
    launch(args.ft_path, args.save_file_path, "tp_result", launch_cmd)
    
    #TP
    launch_cmd = f"mpirun -n 4 --allow-run-as-root python " + argument_cmd + " --tensor_para_size 4 --pipeline_para_size 1"
    launch(args.ft_path, args.save_file_path, "4tp_result", launch_cmd)
    
    #TP
    launch_cmd = f"mpirun -n 8 --allow-run-as-root python " + argument_cmd + " --tensor_para_size 8 --pipeline_para_size 1"
    launch(args.ft_path, args.save_file_path, "8tp_result", launch_cmd)