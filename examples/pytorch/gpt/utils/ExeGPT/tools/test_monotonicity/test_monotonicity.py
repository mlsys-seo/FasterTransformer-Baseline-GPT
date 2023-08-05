import os
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm



debug=False

def debug_print(text):
    if debug:
        pprint(text)


def find_first_greater_index(arr, reference_value, start_index):
    greater_indices = np.where(arr[start_index+1:] >= reference_value)[0]
    
    if len(greater_indices) > 0:
        return greater_indices[0] + start_index
    else:
        return len(arr)


# local min
# def get_nonmonotonic_indices(arr, margin=0):
#     mark_arr = np.zeros(len(arr))
#     # pprint(mark_arr)

#     start_idx = 0
#     while start_idx == len(arr):
#         local_min_value = np.min(arr[start_idx:])
#         local_min_idx = np.argmin(arr[start_idx:])

#         if start_idx + 1 == local_min_idx:
#             start_idx += 1
#             continue
#         elif local_min_value >= arr[start_idx] - margin:
#             start_idx += 1
#             continue
#         else:
#             end_idx = local_min_idx
#             mark_arr[start_idx:end_idx+1] = 1
#             start_idx = end_idx + 1
#             continue
#     # pprint(arr)
#     # pprint(mark_arr)
#     return mark_arr.nonzero()[0]
    
def get_nonmonotonic_indices(arr, margin=0):
    dx_below_zero_idx = np.asarray(np.diff(arr, prepend=arr[0]) < 0-margin).nonzero()[0]

    dx_upper_zero_idx = np.asarray(np.diff(arr, prepend=arr[0]) > 0+margin).nonzero()[0]
    
    if len(dx_below_zero_idx) < len(dx_upper_zero_idx):
        return dx_below_zero_idx
    else:
        return dx_upper_zero_idx
    # return dx_below_zero_idx

# def get_nonmonotonic_indices(arr, margin=0):
#     dx_below_zero_idx = np.asarray(np.diff(arr, prepend=arr[0]) < 0).nonzero()[0]

#     arr_two = arr[:len(arr)-1]
#     arr_two = np.insert(arr_two, 0, 0)
#     diff_np = arr - arr_two
#     dx_below_zero_idx = np.asarray(diff_np+margin < 0).nonzero()[0]
#     return dx_below_zero_idx

#     # dx_below_zero_idx = np.asarray(np.diff(arr, prepend=arr[0]) < 0).nonzero()[0]
    
#     debug_print(f"dx_below_zero_idx: {dx_below_zero_idx}")
    
#     nonmonotonic_idx_list = []
#     for search_idx, start_idx in enumerate(dx_below_zero_idx):
#         debug_print("----")
#         if len(nonmonotonic_idx_list):
#             if start_idx <= nonmonotonic_idx_list[-1]:
#                 continue

#         local_maxima_value = arr[start_idx-1]

#         end_idx = find_first_greater_index(arr, local_maxima_value, int(start_idx))


#         if end_idx is not None:
#             debug_print(f"start_idx: {start_idx}, end_idx: {end_idx}")
#             debug_print(f"local_maxima_value: {local_maxima_value}, margin: {margin}")
#             apply_margin_indices = np.where(arr[start_idx:end_idx+1] < local_maxima_value-margin)[0] + start_idx
#             debug_print(f"apply_margin_indices: {apply_margin_indices}")
#             nonmonotonic_idx_list = nonmonotonic_idx_list + apply_margin_indices.tolist()

#     debug_print("\n\n")
#     return nonmonotonic_idx_list


def get_df_monotonic(df, x_label, y_label, col_name, margin=0):
    df = df.sort_values(x_label)
    nonmonotonic_indices = get_nonmonotonic_indices(df[y_label].to_numpy(), margin)
    debug_print(nonmonotonic_indices)
    df[col_name] = np.arange(len(df))
    df[col_name] = df[col_name].isin(nonmonotonic_indices)
    return df


def get_non_monotonicity(df, x_labels, y_label, margin=0, is_draw_plot=False):
    
    case_dic = { x_label: pd.DataFrame({f'{x_label}': df[x_label].unique()}) for x_label in x_labels }
    debug_print(f"x_labels: {x_labels}")
    result_df = df.copy()

    num_nonmonotonic_dicts = { target_label: {} for target_label in x_labels }
    per_target_df_list = []
    for target_label in x_labels:
        untarget_labels = [ label for label in df.columns if label not in [target_label, y_label] ]
        case_df = df[untarget_labels].drop_duplicates().reset_index(drop=True)
        # pprint(case_df)
        # input()
        col_name = f'{target_label}_nonmonotonic'
        
        debug_print(f'target_label: {target_label}')
        debug_print(f'untarget_labels: {untarget_labels}')
        debug_print(f'len_case_of: {len(case_df)}')
        

        target_result_df_list = []
        for case_ids, case in tqdm(case_df.iterrows(), desc=f"-- target label: {target_label}"):
            # print(f"\n{case}")
            filtered_df = pd.merge(df, pd.DataFrame(case).transpose(), on=untarget_labels, how='inner')

            case_result_df = get_df_monotonic(filtered_df, target_label, y_label, col_name, margin)

            debug_print("\n\ncase_result_df:")
            debug_print(case_result_df)
            target_result_df_list.append(case_result_df)
            num_nonmonotonic_dicts[target_label][case_ids] = case_result_df[col_name].sum().item()
        target_result_df = pd.concat(target_result_df_list)
        per_target_df_list.append(target_result_df)

        if is_draw_plot:
            target_row_idx = min(num_nonmonotonic_dicts[target_label], key=num_nonmonotonic_dicts[target_label].get)
            print(target_row_idx)
            print(type(target_row_idx))
            # if isinstance(target_row_idx, tuple):
            #     target_row_idx = target_row_idx[0]
            pprint(pd.DataFrame(case_df.loc[target_row_idx, :]).transpose())
            filtered_df = pd.merge(target_result_df, pd.DataFrame(case_df.loc[target_row_idx, :]).transpose(), on=untarget_labels, how='inner')
            filtered_df = filtered_df.sort_values(target_label)
            
            title = f"target: {target_label},"
            for untarget_label in untarget_labels:
                title += f"{untarget_label}: {case_df.loc[target_row_idx, untarget_label].item()},"
            fig = filtered_df.plot(x=target_label, y=y_label, kind='line', title=title).get_figure()
            fig.savefig(f"./results/{title}.png")

        
    
    merge_on_labels = untarget_labels + [target_label, y_label]
    for target_df in per_target_df_list:
        result_df = result_df.merge(target_df.dropna(), on=merge_on_labels, how='left')
    return result_df

def run_test(name, x_labels, y_label, margins, is_draw_plot):
    data_pd = pd.read_csv(f'{name}.csv', index_col=0)

    debug_print(data_pd)
    debug_print(x_labels)
    debug_print(y_label)

    tp_list = [1, 2, 4, 8]
    pp_list = [16, 8, 4, 2]

    pair_pd = data_pd[['tp', 'pp']].drop_duplicates()

    summary_dicts = []
    for margin in margins:
        print(f"-- margin: {margin}")
        result_pd = get_non_monotonicity(data_pd, x_labels, y_label, margin, is_draw_plot)

        col_names = [ f'{target_label}_nonmonotonic' for target_label in x_labels ]
        result_pd.loc[:, 'nonmonotonicity_point'] = result_pd.loc[:, col_names].sum(axis=1)
        # result_pd.loc[:, 'nonmonotonicity_point'] = result_pd['encoder_frequency_nonmonotonic']
        
        for tp, pp in zip(pair_pd['tp'], pair_pd['pp']):
            temp_df = result_pd.loc[(result_pd['tp'] == tp) & (result_pd['pp'] == pp),].copy()
            summary_dicts.append({
                "PP": pp,
                "TP": tp,
                "margin": margin,
                "num_non-monotonic-point": sum(temp_df['nonmonotonicity_point']),
                "num_all": len(result_pd),
                "non-monotonic-ratio": round(sum(temp_df['nonmonotonicity_point'])/len(result_pd)*100, 2),
            })
            # print(f"PP:{pp} \t TP:{tp} \t FOR BOUND {margin} \t {sum(temp_df['nonmonotonicity_point'])} \t {len(data_pd)} \t {round(sum(temp_df['nonmonotonicity_point'])/len(data_pd), 2)*100} %", )
        result_pd.to_csv(f'./results/{name}_{margin}_result.csv')
    summary_pd = pd.DataFrame(summary_dicts)
    summary_pd.to_csv(f'./results/{name}_summary.csv')
    return summary_pd

if __name__ == "__main__":

    # # test find_first_greater_index()
    # arr = np.array([1, 3, 5, 2, 4])
    # assert find_first_greater_index(arr, 3, 2), 4

    # # test get_nonmonotonic_indices()
    # test_values = np.array([1.4, 1.5, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.7, 1.9])
    # assert get_nonmonotonic_indices(test_values, margin=0.1), [4, 5, 6, 7, 8, 9]



    # x_labels = ['a', 'b', 'c']
    # y_label = 'y'

    # get_non_monotonicity(pd.DataFrame({
    #     'a': [0,1,2,3,4],
    #     'b': [1,1,1,1,2],
    #     'c': [2,2,2,2,2],
    #     'y': [3,4,5,3,3]
    # }), x_labels, y_label)

    #########################################


    os.makedirs("./results", exist_ok=True)

    # rra
    print("\n:: START RRA ::")
    x_labels = ['decoder_batch_size', 'encoder_frequency']
    y_label = 'latency'
    margins = [240, 720, 1200]

    summary_pd = run_test('rra', x_labels, y_label, margins, is_draw_plot=True)
    pprint(summary_pd)
    print(":: END RRA ::")

    # waa
    print("\n:: START WAA ::")
    x_labels = ['micro_batch_num', 'encoder_batch_size']
    y_label = 'latency'
    margins = [240, 720, 1200]

    summary_pd = run_test('waa', x_labels, y_label, margins, is_draw_plot=True)
    pprint(summary_pd)
    print(":: END WAA ::")