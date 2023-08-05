import json
from typing import List
import os
from .dtype import TPType

_PROF_RESULT_JSON = None

def load_prof_result(file_path: str, tp=None) -> None:
    global _PROF_RESULT_JSON
    _PROF_RESULT_JSON = {}
    
    if tp is None:
        single_result_file = os.path.join(file_path, "single_result.json")
        with open(single_result_file, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.none}"] = json.load(json_file)

        tp_result_file_2 = os.path.join(file_path, "tp_result.json")
        with open(tp_result_file_2, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp2}"] = json.load(json_file)

        tp_result_file_4 = os.path.join(file_path, "4tp_result.json")
        with open(tp_result_file_4, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp4}"] = json.load(json_file)

        tp_result_file_8 = os.path.join(file_path, "8tp_result.json")
        with open(tp_result_file_8, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp8}"] = json.load(json_file)

    if tp == 0:
        single_result_file = os.path.join(file_path, "single_result.json")
        with open(single_result_file, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.none}"] = json.load(json_file)

    elif tp == 2:
        tp_result_file_2 = os.path.join(file_path, "tp_result.json")
        with open(tp_result_file_2, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp2}"] = json.load(json_file)

    elif tp == 4:
        tp_result_file_4 = os.path.join(file_path, "4tp_result.json")
        with open(tp_result_file_4, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp4}"] = json.load(json_file)
    
    elif tp == 8:    
        tp_result_file_8 = os.path.join(file_path, "8tp_result.json")
        with open(tp_result_file_8, "r") as json_file:
            _PROF_RESULT_JSON[f"{TPType.tp8}"] = json.load(json_file)


    # attn_result = os.path.join(file_path, "attn_result.json")
    # kv_result = os.path.join(file_path, "kv_result.json")
    # tp_attn_result = os.path.join(file_path, "tp_attn_result.json")
    # tp_kv_result = os.path.join(file_path, "tp_kv_result.json")
    
    
    # with open(attn_result, "r") as json_file:
    #     _PROF_RESULT_JSON["attn"] = json.load(json_file)
    
    # with open(kv_result, "r") as json_file:
    #     _PROF_RESULT_JSON["kv"] = json.load(json_file)
        
    # with open(tp_attn_result, "r") as json_file:
    #     _PROF_RESULT_JSON["tp_attn"] = json.load(json_file)

    # with open(tp_kv_result, "r") as json_file:
    #     _PROF_RESULT_JSON["tp_kv"] = json.load(json_file)
        

def get_layer_profile_result(tp_type: TPType=TPType.none):
    global _PROF_RESULT_JSON
    assert _PROF_RESULT_JSON is not None, "Profile_result should be load first"
    
    return _PROF_RESULT_JSON[f"{tp_type}"]
        


# def get_attn_profile_result(tp_type: TPType=TPType.none):
#     global _PROF_RESULT_JSON
#     assert _PROF_RESULT_JSON is not None, "Profile_result should be load first"
    
#     if tp_type != TPType.none:
#         return _PROF_RESULT_JSON[f"attn"]
#     return _PROF_RESULT_JSON[f"tp_attn"]


def get_kv_profile_result(tp_type: TPType=TPType.none):
    global _PROF_RESULT_JSON
    assert _PROF_RESULT_JSON is not None, "Profile_result should be load first"
    
    return _PROF_RESULT_JSON[f"{tp_type}"]


def get_profiled_batch(tp_type: TPType=TPType.none) -> List[int]:
    global _PROF_RESULT_JSON
    try:
        return list(map(int, _PROF_RESULT_JSON[f"{tp_type}"].keys()))
    except KeyError:
        return None

def check_batch_is_profiled(batch: int, tp_type: TPType=TPType.none) -> int:
    profiled_batch = get_profiled_batch(tp_type)
    if profiled_batch is None:
        return None
    if batch not in profiled_batch:
        return False
    return True


def get_layer_profiled_data(batch_size: int, is_decoder: bool=False, tp_type: TPType=TPType.none):
    profiled_data = get_layer_profile_result(tp_type)
    try:
        return profiled_data[str(batch_size)]['Encoder' if not is_decoder else 'Decoder']
    except KeyError:
        return None

# def get_attn_profiled_data(batch_size: int, is_cross_attn: bool=False, tp_type: TPType=TPType.none):
#     profiled_data = get_attn_profile_result(tp_type)
#     return profiled_data[str(batch_size)]['Self_Attention' if not is_cross_attn else 'Cross_Attention']

def get_kv_profiled_data(batch_size: int, tp_type: TPType=TPType.none):
    profiled_data = get_layer_profile_result(tp_type)
    try:
        return profiled_data[str(batch_size)]['KV']
    except KeyError:
        return 0.


def get_accumulate_point(batch_size: int, tp_type: TPType=TPType.none):
    profiled_batch = get_profiled_batch(tp_type)
    larger_batch = 0
    smaller_batch = 0
    if batch_size > profiled_batch[-1]:
        larger_batch = profiled_batch[-1]
        smaller_batch = profiled_batch[-2]

    elif batch_size < profiled_batch[0]:
        larger_batch = profiled_batch[1]
        smaller_batch = profiled_batch[0]

    else:
        for idx, batch in enumerate(profiled_batch):
            if batch_size < batch:
                larger_batch = batch
                smaller_batch = profiled_batch[idx - 1]
                break
        
    return larger_batch, smaller_batch