from .dtype import (
    c1_config,
    c2_config,
    config,
    PerfEstim,
    GPUPart,
    TPType
)
 
from math import ceil, floor
from .profile_analyzer import query_latency


def gen_exec_map_c1(config: c1_config) -> list:
    encoder_count = config.get_encoder_device_count()
    decoder_count = config.get_decoder_device_count()
    
    encoder_only_count = config.n_vgpus - decoder_count
    
    # TODO: Add encoder decoder bubble
    
    exec_map = [[(0, 0, 0) for _ in range(node_idx + encoder_only_count)] for node_idx in range(decoder_count)]
    if decoder_count >= config.decoder_mb_count:
        iter_cnt = (config.n_decoder_seq_len - 1) * decoder_count + config.decoder_mb_count
    else:
        iter_cnt = config.n_decoder_seq_len * config.decoder_mb_count
    
    for idx, node in enumerate(exec_map):
        encoder_batch = config.encoder_sliced_batch_size if config.gpu_topo[idx + encoder_count - 1].is_encoder() else 0
        decoder_batch = config.decoder_batch_size
        kv_batch = config.encoder_sliced_batch_size if config.n_vgpus < encoder_count + decoder_count else config.encoder_batch_size
        part = (encoder_batch, decoder_batch, kv_batch)
        node += [part for _ in range(iter_cnt)]
                
        for _ in range(len(exec_map) - idx - 1):
            node.append((0, 0, 0))
    
    for idx in range(encoder_only_count):
        exec_list = [(0, 0, 0) for _ in range(len(exec_map[0]) - 1)]
        exec_list.insert(idx, (config.encoder_batch_size, 0, 0))
        exec_map.insert(idx, exec_list)
    
    # if config.decoder_mb_count != 0:
    throughput_exec_map = [[(0, config.decoder_batch_size, config.encoder_batch_size) for __ in  range(config.decoder_mb_count)] for _ in range(config.n_vgpus)]
    # else:
    #     throughput_exec_map = [[(0, config.decoder_batch_size, config.encoder_batch_size)] for _ in range(config.n_vgpus)]

    return exec_map, throughput_exec_map, config.encoder_batch_size


def gen_exec_map_c2(config: c2_config) -> list:
    latency_exec_map = [[(0, 0, 0, 0) for _ in range(node_idx)] for node_idx in range(config.n_vgpus)]
    throughpuyt_exec_map = [[(0, 0, 0, 0) for _ in range(node_idx)] for node_idx in range(config.n_vgpus)]
    
    # latency_exec_tmp = []
    # throughput_exec_tmp = []
    window_exec_map = [(config.encoder_batch_size, 0, 0, 0 if idx != config.n_vgpus-1 else config.encoder_batch_size) for idx in range(config.n_vgpus)]
    # window_exec_map = [(ceil(config.encoder_batch_size / config.mb), 0, 0, 0 if idx != config.n_vgpus-1 else config.encoder_batch_size) for idx in range(config.n_vgpus * config.mb)]
    # window_exec_map = [(2, 0, 0, 0 if idx != config.n_vgpus-1 else config.encoder_batch_size) for idx in range(ceil(config.encoder_batch_size/2))]
    for idx, decoder_batch in enumerate(zip(config.decoder_batch_list, config.embd_batch_list)):
        for idx in range(config.n_vgpus):
            window_exec_map.append((0, decoder_batch[0], config.encoder_batch_size if idx == 0 else 0, decoder_batch[0] if idx == config.n_vgpus - 1 else 0))
    
    runned_decoder = (config.n_decoder_seq_len // len(config.decoder_batch_list)) * len(config.decoder_batch_list)
    left_decoder = config.n_decoder_seq_len - runned_decoder
    
    latency_exec_tmp = window_exec_map * (config.n_decoder_seq_len // len(config.decoder_batch_list))
    if left_decoder != 0:
        latency_exec_tmp += window_exec_map[:(left_decoder + 1) * config.n_vgpus]
    
    throughput_exec_tmp = window_exec_map
    
    for idx in range(len(latency_exec_map)):
        latency_exec_map[idx] += latency_exec_tmp
        if idx != 0:
            throughpuyt_exec_map[idx] += throughput_exec_tmp[:-idx]
        else:
            throughpuyt_exec_map[idx] += throughput_exec_tmp
        for _ in range(config.n_vgpus - idx - 1):
            latency_exec_map[idx].append((0, 0, 0))
            # throughpuyt_exec_map[idx].append((0, 0, 0))
    
    # def query(batch, is_decoder):
    #     return query_latency(80, batch, is_decoder, TPType.none)
    
    # for a in latency_exec_map[0]:
    #     if a[0] != 0:
    #         print(f"Encoder\t{a[0]}\t{query(a[0], False)}")
    #     elif a[1] != 0:
    #         print(f"Decoder\t{a[1]}\t{query(a[1], True)}")
     
    # print("================================================================")       
            
    # for a in latency_exec_map[1]:
    #     if a[0] != 0:
    #         print(f"Encoder\t{a[0]}\t{query(a[0], False)}")
    #     elif a[1] != 0:
    #         print(f"Decoder\t{a[1]}\t{query(a[1], True)}")
    # print(latency_exec_map)
    # print(throughpuyt_exec_map)
    
    return latency_exec_map, throughpuyt_exec_map, config.encoder_batch_size


def convert_for_estimate(exec_map: list) -> list:
    exec_list = []
    for idx in range(len(exec_map[0])):
        tmp = []
        for node in exec_map:
            tmp.append(node[idx])
        exec_list.append(tmp)
            
    return exec_list
        

def get_bottleneck_latency(config: config, exec_list: list) -> float:
    # node_time = []
    # for node, topo in zip(exec_list, config.gpu_topo):
    #     time_ = query_latency(topo.encoder, node[0], False, topo.tp_type) + \
    #         query_latency(topo.decoder, node[1], True, topo.tp_type, node[2])
    #     node_time.append(time_)
    # return max(node_time)
    
    
    return max(list(map((lambda node, topo: \
        query_latency(topo.encoder, node[0], False, topo.tp_type) + \
        query_latency(topo.decoder, node[1], True, topo.tp_type, node[2])), \
            exec_list, config.gpu_topo)))


def estimate_latency_throughput(config: config, latency_exec_map, throughput_exec_map, consumed_batch) -> float:
    latency_exec_list = convert_for_estimate(latency_exec_map)
    throughput_exec_list = convert_for_estimate(throughput_exec_map)
    
    latency = 0.
    for slice in latency_exec_list:
        a = get_bottleneck_latency(config, slice)
        latency += a
    
    throughput = 0.
    for slice in throughput_exec_list:
        a = get_bottleneck_latency(config, slice)
        throughput += a

    throughput = consumed_batch / throughput
    
    return PerfEstim(latency, throughput)


def estimate_laytency_throughput(config: config) -> PerfEstim:
    if isinstance(config, c1_config):
        latency_exec_map, throughput_exec_map, consumed_batch = gen_exec_map_c1(config)
    else:
        latency_exec_map, throughput_exec_map, consumed_batch = gen_exec_map_c2(config)
    result = estimate_latency_throughput(config, latency_exec_map, throughput_exec_map, consumed_batch)
        
    return result


def optimize_staging(config: config):
    create_gpu_topo(config)
    update_for_tp(config)
    
    if isinstance(config, c1_config):
        send_func = send_layer_c1
    else:
        # send_func = send_layer_c2
        return
    
    for _ in range(config.n_vgpus * config.n_vgpus * config.n_vgpus):
        latency = get_all_node_latency(config)
        longest_node = latency.index(max(latency))
        
        # if check_is_in_bound(config, longest_node):
        #     break
        
        shortest_node = latency.index(min(latency))
        src = longest_node
        for step in range(1, abs(longest_node - shortest_node)+1):
            if longest_node > shortest_node:
                tar = longest_node - step
            else:
                tar = longest_node + step
            
            send_func(config, src, tar)
            src = tar


def Performance_Estim(config: config) -> PerfEstim:
    if len(config.gpu_topo) > 0:
        # print(config.gpu_topo)
        # print(get_all_node_latency(config))
        return estimate_laytency_throughput(config)
    
    optimize_staging(config)
    # print(config.gpu_topo)
    # print(get_all_node_latency(config))
    # print(config.encoder_batch_size)
    # print(config.decoder_batch_size, config.decoder_mb_count)
    
    return estimate_laytency_throughput(config)

def get_all_node_latency(config: config) -> list:
    if isinstance(config, c1_config):
        return get_all_node_latency_c1(config)
    else:
        return get_all_node_latency_c2(config)

def get_all_node_latency_c1(config: c1_config) -> list:
    return list(map((lambda node: \
        query_latency(node.encoder, config.encoder_batch_size, False, node.tp_type) + \
        query_latency(node.decoder, config.encoder_batch_size * config.n_decoder_seq_len, True, node.tp_type, config.encoder_batch_size)), config.gpu_topo))

def get_all_node_latency_c2(config: c2_config) -> list:
    return list(map((lambda node: \
        query_latency(node.encoder, config.encoder_batch_size, False, node.tp_type) + \
        query_latency(node.decoder, config.encoder_batch_size, True, node.tp_type)), config.gpu_topo))
    
def send_layer_c1(config: c1_config, src: int, tar: int) -> None:
    src_node = config.gpu_topo[src]
    tar_node = config.gpu_topo[tar]
    if src_node.is_decoder() and not src_node.is_encoder():
        src_node.decoder -= 1
        tar_node.decoder += 1
    elif not src_node.is_decoder() and src_node.is_encoder():
        src_node.encoder -= 1
        tar_node.encoder += 1
    else:
        if tar_node.is_encoder() and not tar_node.is_decoder():
            src_node.encoder -= 1
            tar_node.encoder += 1
        else:
            src_node.decoder -= 1
            tar_node.decoder += 1
            

def update_for_tp(config: config) -> None:    
    if not config.is_tp():
        return
    
    for _ in range(config.tp_gpu_num):
        encoder_runtime = get_single_layer_runtime(config, False)
        decoder_runtime = get_single_layer_runtime(config, True) * config.n_decoder_seq_len
        
        not_tp_list = []
        for node in config.gpu_topo:
            if node.tp_type == TPType.none:
                not_tp_list.append(node)

        
        if encoder_runtime > decoder_runtime:
            head = not_tp_list[0]
            r_node = not_tp_list[1]
        else:
            head = not_tp_list[-1]
            r_node = not_tp_list[-2]
            
        head += r_node
        head.tp_type = TPType.tp
        config.gpu_topo.remove(r_node)


def split_layer_per_node(n_layers, split_gpus, n_gpus):
    layer_list = []
    
    layer_cnt = n_layers
    layer_per_node = floor(n_layers / split_gpus)
    
    for _ in range(n_gpus):
        if layer_cnt >= layer_per_node:
            layer_list.append(layer_per_node)
            layer_cnt -= layer_per_node
        else:
            layer_list.append(0)
            
    for idx in range(layer_cnt):
        layer_list[idx] += 1
    
    return layer_list    


def create_gpu_topo(config: config) -> None:
    if isinstance(config, c1_config):
        
        encoder_gpu_cnt = ceil(config.n_gpus / 2)
        decoder_gpu_cnt = config.n_gpus - encoder_gpu_cnt
        
        encoder_list = split_layer_per_node(config.n_layers, encoder_gpu_cnt, config.n_gpus)
        decoder_list = split_layer_per_node(config.n_layers, decoder_gpu_cnt, config.n_gpus)
        
        decoder_list.reverse()
        
        for e, d in zip(encoder_list, decoder_list):
            config.gpu_topo.append(GPUPart(e, d))

    else:
        layer_list = split_layer_per_node(config.n_layers, config.n_gpus, config.n_gpus)
        
        for layer_cnt in layer_list:
            config.gpu_topo.append(GPUPart(layer_cnt, layer_cnt))

def get_single_layer_runtime(config: config, is_decoder: bool=False) -> float:
    if is_decoder:
        return query_latency(1, config.decoder_batch_size, is_decoder=True, kv_batch=config.encoder_batch_size)
    else:
        return query_latency(1, config.encoder_batch_size, is_decoder=False)

# def check_is_in_bound(config: config, node: int) -> bool:       
#     latency = get_all_node_latency(config)
#     encoder_layer_runtime = get_single_layer_runtime(config, False)
#     decoder_layer_runtime = get_single_layer_runtime(config, True) * config.decoder_mb_count
#     mean_laytency = sum(latency) / len(latency)
#     bound = mean_laytency + max(encoder_layer_runtime, decoder_layer_runtime)
    
#     return bound >= latency[node]

# def send_layer_c2(config: c1_config, src: int, tar: int) -> None:
#     latency = get_all_node_latency(config)
    
#     src_node = config.gpu_topo[src]
#     tar_node = config.gpu_topo[tar]
    
#     encoder_layer_runtime = get_single_layer_runtime(config, False)
#     decoder_layer_runtime = get_single_layer_runtime(config, True)
    
#     diff_latency = abs(latency[src] - latency[tar])
    
#     if abs(diff_latency - encoder_layer_runtime) > abs(diff_latency - decoder_layer_runtime):
#         src_node.encoder -= 1
#         tar_node.encoder += 1
#     else:
#         src_node.decoder -= 1
#         tar_node.decoder += 1



