import subprocess
import argparse
import os


def launch(save_path, file_name, cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    print(p.stdout.decode('UTF-8'))
    with open(os.path.join(save_path, f"{file_name}.json"), 'w') as f:
        f.write(p.stdout.decode('UTF-8'))
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--kv-channels", type=int)
    parser.add_argument("--ffn-hidden-size", type=int)
    parser.add_argument("--encoder-seq-length", type=int)
    parser.add_argument("--decoder-seq-length", type=int)
    parser.add_argument("--decoder-max-seq-length", type=int)
    parser.add_argument("--start-batch-size", type=int)
    parser.add_argument("--end-batch-size", type=int)
    parser.add_argument("--batch-size-hop", type=int)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--mask-prob", type=float)
    parser.add_argument("--layernorm-epsilon", type=float)
    parser.add_argument("--sample-iters", type=int)
    parser.add_argument("--save-file-path", type=str)
    parser.add_argument("--megatron-path", type=str)
    parser.add_argument("--padded-vocab-size", type=int)
    parser.add_argument('--fp16', action='store_true',
                        default=False)
    parser.add_argument('--overlap-tensor-model-parallel', action='store_true',
                        default=False)

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    argument_cmd = f"{os.path.join(args.megatron_path, 'pretrain_t5.py')} \
                --num-layers 1 \
                --start-batch-size {args.start_batch_size} \
                --end-batch-size {args.end_batch_size} \
                --batch-size-hop {args.batch_size_hop} \
                --hidden-size {args.hidden_size} \
                --num-attention-heads {args.num_attention_heads} \
                --kv-channels {args.kv_channels} \
                --ffn-hidden-size {args.ffn_hidden_size} \
                --encoder-seq-length {args.encoder_seq_length} \
                --decoder-seq-length {args.decoder_seq_length} \
                --max-position-embeddings {args.max_position_embeddings} \
                --padded-vocab-size {args.padded_vocab_size} \
                --eval-iters 1 \
                --mask-prob {args.mask_prob} \
                --no-bias-gelu-fusion \
                --layernorm-epsilon {args.layernorm_epsilon} \
                --sample-iters {args.sample_iters}"
                
    kv_argument_cmd = f"{os.path.join(args.megatron_path, 'kv_test.py')} \
                --hidden-size {args.hidden_size} \
                --num-attention-heads {args.num_attention_heads} \
                --kv-channels {args.kv_channels} \
                --start-batch-size {args.start_batch_size} \
                --end-batch-size {args.end_batch_size} \
                --batch-size-hop {args.batch_size_hop} \
                --encoder-seq-length {args.encoder_seq_length} \
                --sample-iters {args.sample_iters}"
    
    attn_argument_cmd = f"{os.path.join(args.megatron_path, 'attn_test.py')} \
                --num-attention-heads {args.num_attention_heads} \
                --kv-channels {args.kv_channels} \
                --start-batch-size {args.start_batch_size} \
                --end-batch-size {args.end_batch_size} \
                --batch-size-hop {args.batch_size_hop} \
                --encoder-seq-length {args.encoder_seq_length} \
                --decoder-max-seq-length 128 \
                --sample-iters {args.sample_iters} \
                --t5-attn-profile"
                
    tp_kv_argument_cmd = f"{os.path.join(args.megatron_path, 'kv_test.py')} \
                --hidden-size {args.hidden_size} \
                --num-attention-heads {args.num_attention_heads//2} \
                --kv-channels {args.kv_channels} \
                --start-batch-size {args.start_batch_size} \
                --end-batch-size {args.end_batch_size} \
                --batch-size-hop {args.batch_size_hop} \
                --encoder-seq-length {args.encoder_seq_length} \
                --sample-iters {args.sample_iters}"
    
    tp_attn_argument_cmd = f"{os.path.join(args.megatron_path, 'attn_test.py')} \
                --num-attention-heads {args.num_attention_heads//2} \
                --kv-channels {args.kv_channels} \
                --start-batch-size {args.start_batch_size} \
                --end-batch-size {args.end_batch_size} \
                --batch-size-hop {args.batch_size_hop} \
                --encoder-seq-length {args.encoder_seq_length} \
                --decoder-max-seq-length {args.decoder_max_seq_length} \
                --sample-iters {args.sample_iters} \
                --t5-attn-profile"

    if args.fp16:
        argument_cmd += " --fp16"
        kv_argument_cmd += " --fp16"
        attn_argument_cmd += " --fp16"
        tp_kv_argument_cmd += " --fp16"
        tp_attn_argument_cmd += " --fp16"

    #Single GPU
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        argument_cmd
    launch(args.save_file_path, "single_result", launch_cmd)
    
    #TP
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        argument_cmd + " --tensor-model-parallel-size 2"
    launch(args.save_file_path, "tp_result", launch_cmd)
    
    #CTP
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        argument_cmd + " --tensor-model-parallel-size 2 --overlap-tensor-model-parallel"
    launch(args.save_file_path, "ctp_result", launch_cmd)
    
    #KV
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        kv_argument_cmd
    launch(args.save_file_path, "kv_result", launch_cmd)
    
    #Attention
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        attn_argument_cmd
    launch(args.save_file_path, "attn_result", launch_cmd)
    
    #TP KV
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        tp_kv_argument_cmd
    launch(args.save_file_path, "tp_kv_result", launch_cmd)
    
    #TP Attention
    launch_cmd = f"NCCL_ASYNC_ERROR_HANDLING=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6004 " + \
        tp_attn_argument_cmd
    launch(args.save_file_path, "tp_attn_result", launch_cmd)