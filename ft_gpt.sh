START_BATCH=8
END_BATCH=8
BATCH_HOP=8
E_SEQ=128
D_SEQ=32

mkdir -p /workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40

# 4GPU
mpirun -n 4 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/1-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 1 \
    --pipeline_para_size 4 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_4GPU_4PP.json"
    
mpirun -n 4 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/2-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 2 \
    --pipeline_para_size 2 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_4GPU_2PP_2TP.json"

mpirun -n 4 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/4-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 4 \
    --pipeline_para_size 1 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_4GPU_4TP.json"

#8GPU
mpirun -n 8 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/1-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 1 \
    --pipeline_para_size 8 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_8GPU_8PP.json"
    
mpirun -n 8 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/2-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 2 \
    --pipeline_para_size 4 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_8GPU_4PP_2TP.json"

mpirun -n 8 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/4-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 4 \
    --pipeline_para_size 2 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_8GPU_2PP_4TP.json"

mpirun -n 8 --allow-run-as-root \
python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path "/workspace/BBB/model/FT/GPT-175B/8-gpu" \
    --data_type fp16 \
    --input_len $E_SEQ \
    --output_len $D_SEQ \
    --max_seq_len $(($E_SEQ+$D_SEQ+1)) \
    --profile_iters 10 \
    --start_batch_size $START_BATCH \
    --end_batch_size $END_BATCH \
    --batch_size_hop $BATCH_HOP \
    --layer_num 96 \
    --tensor_para_size 8 \
    --pipeline_para_size 1 \
    --save_path "/workspace/BBB-Result/GPT/175b/${E_SEQ}-${D_SEQ}/a40/base_8GPU_8TP.json"