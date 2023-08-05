from .global_vars import check_batch_is_profiled, get_layer_profiled_data, get_kv_profiled_data, get_accumulate_point
from typing import Tuple
from .dtype import TPType


def accumulate_latency(p_1: Tuple[int, int], p_2:Tuple[int, int], x_t: int) -> float:
    
    (x_1, y_1) = p_1
    (x_2, y_2) = p_2
    
    a = abs(y_1 - y_2) / (x_1 - x_2)
    b = y_1 - (a * x_1)
    return a * x_t + b


def query_latency(layer_cnt: int, batch_size: int, 
                  is_decoder: bool=False, tp_type: TPType=TPType.none,
                  kv_batch: int=None) -> float:
    if layer_cnt == 0 or batch_size == 0:
        return 0.

    check = check_batch_is_profiled(batch_size, tp_type)    
    # if check is None:
    #     return None
    
    if not check:
        p_1_x, p_2_x = get_accumulate_point(batch_size, tp_type)
        p_1_y = get_layer_profiled_data(p_1_x, is_decoder, tp_type)
        p_2_y = get_layer_profiled_data(p_2_x, is_decoder, tp_type)
        layer_time = accumulate_latency((p_1_x, p_1_y), (p_2_x, p_2_y), batch_size)
        if layer_time is None:
            return None
        # if is_decoder:
        #     p_1_y = get_attn_profiled_data(p_1_x, False, tp_type)
        #     p_2_y = get_attn_profiled_data(p_2_x, False, tp_type)
        #     layer_time += accumulate_latency((p_1_x, p_1_y), (p_2_x, p_2_y), batch_size)
            
            
        #     p_1_y = get_attn_profiled_data(p_1_x, True, tp_type)
        #     p_2_y = get_attn_profiled_data(p_2_x, True, tp_type)
        #     layer_time += accumulate_latency((p_1_x, p_1_y), (p_2_x, p_2_y), batch_size)
    
    else:
        layer_time = get_layer_profiled_data(batch_size, is_decoder, tp_type)
        if layer_time is None:
            print(batch_size)
        # if is_decoder:
        #     layer_time += get_attn_profiled_data(batch_size, False, tp_type)
        #     layer_time += get_attn_profiled_data(batch_size, True, tp_type)
    
    if is_decoder and kv_batch is not None:
        if not check_batch_is_profiled(kv_batch, tp_type):
            p_1_x, p_2_x = get_accumulate_point(kv_batch, tp_type)
            p_1_y = get_kv_profiled_data(p_1_x, tp_type)
            p_2_y = get_kv_profiled_data(p_2_x, tp_type)
            layer_time += accumulate_latency((p_1_x, p_1_y), (p_2_x, p_2_y), kv_batch)
        else:
            layer_time += get_kv_profiled_data(kv_batch, tp_type)
    # if is_decoder and batch_size == 32:
        # print(layer_time)
    # return layer_time
    
    # if not is_decoder:
    #     layer_cnt *= 0.7
    
    return layer_time * layer_cnt
    # return layer_time / 16 * 24
