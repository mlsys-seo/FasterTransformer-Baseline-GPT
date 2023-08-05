import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
batch_size = 4

generator = pipeline('text-generation', model='facebook/opt-2.7b',
                    batch_size=batch_size,
                    device=local_rank)

# GPT같은 Causal LM은 max_new_tokens를 써야 하고,
# T5와 같은 LM은 max_length를 써야 한다.
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True)

audio_filenames = [f"audio_{i}.flac" for i in range(batch_size)]


string = generator(audio_filenames, do_sample=False, num_beams=1, min_length=50, max_new_tokens=1024)

print("")
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)