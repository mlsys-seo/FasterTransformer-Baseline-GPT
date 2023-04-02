PROFILE_NSYS="nsys profile --cuda-graph-trace graph -t cuda,nvtx -o /workspace/profile/test_ft_gpt_tp -w true -f true"

FT_NVTX=ON $PROFILE_NSYS python examples/pytorch/gpt/multi_gpu_gpt_example.py \
    --ckpt_path="models/megatron-models/c-model/GPT-175B/1-gpu" \
    --data_type fp16 \
    --max_batch_size=32 \
    --input_len=32 \
    --output_len=32 \
    --max_seq_len=128
    # --profile_iters 10