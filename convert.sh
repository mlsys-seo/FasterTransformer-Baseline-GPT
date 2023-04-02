export PYTHONPATH=PWD/..:PWD/..:{PYTHONPATH}:3rdparty/Megatron-LM

python examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -head_num 96 \
        -i 3rdparty/Megatron-LM/GPT_SAVE_175B/iter_0000001 \
        -o models/megatron-models/c-model/GPT-175B_TP/ \
        -t_g 1 \
        -i_g 2 \
        --vocab-path /workspace/data/Inference/gpt2-vocab.json \
        --merges-path /workspace/data/Inference/gpt2-merges.txt