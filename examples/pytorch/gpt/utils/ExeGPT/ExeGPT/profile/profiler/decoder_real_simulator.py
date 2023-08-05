from scipy.stats import poisson, uniform
import random

def pois_dist(n, lamb, max_seq):
    assert n <= max_seq
    if max_seq == n:
        return 1 - poisson.cdf(max_seq-1, mu=lamb)
    elif n == 1:
        return 0
    elif n == 2:
        return poisson.cdf(n, mu=lamb)
    else:
        return poisson.pmf(n, mu=lamb)

def unif_dist(n, max_seq):
    assert n <= max_seq
    if max_seq == n:
        return 1 - uniform.cdf(max_seq-1, scale=max_seq)
    elif n == 1:
        return uniform.cdf(n, scale=max_seq)
    else:
        return uniform.pdf(n, scale=max_seq)

class DataObject:

    num_generated_object = 0

    def __init__(self, target_seq_len):

        self.id = DataObject.num_generated_object
        self.target_seq_len = target_seq_len

        self.processed_seq = 0

        DataObject.num_generated_object += 1

    def is_end(self):
        return self.processed_seq == self.target_seq_len

    def process(self):
        self.processed_seq += 1

    def __str__(self):
        return (f"id : {self.id} seq : {self.target_seq_len} p {self.processed_seq}")

def run_process(p_dict):
    for key, data in p_dict.items():
        data.process()

    end_list = []
    for key, data in p_dict.items():
        if data.is_end():
            end_list.append(data.id)

    for data_id in end_list:
        p_dict.pop(data_id)

    return len(end_list)


def run_profile_bsize(encoder_bsize, avg_output_seq_len, window_size):

        ############ Data Preparation ##############
        

        # input_data_list = []
        # max_seq_len = avg_output_seq_len * 2
        # UID = 0
        # sum = 0
        # NUM_FULL_BATCH = 200000
        
        # for target_seq_len in range(1, max_seq_len + 1):
        #     probability = pois_dist(target_seq_len, avg_output_seq_len, max_seq_len)
        #     # probability = unif_dist(target_seq_len, max_seq_len - 1)
        #     num_datas = max(round(probability * NUM_FULL_BATCH), 1)

        #     sum += num_datas
        #     for y in range(num_datas):
        #         dummy_data = DataObject(UID, target_seq_len)
        #         input_data_list.append(dummy_data)
        #         UID += 1
        # # print(sum)
        # random.shuffle(input_data_list)
        
        NUM_FULL_BATCH = 1000000
        NUM_ENCODER_STEP = 3 * avg_output_seq_len // window_size
        max_seq_len = avg_output_seq_len * 2
        
        prob_of_target_seq_len = [ pois_dist(target_seq_len, avg_output_seq_len, max_seq_len) for target_seq_len in range(1, max_seq_len+1) ]
        # prob_of_target_seq_len = [ unif_dist(target_seq_len, max_seq_len - 1) for target_seq_len in range(1, max_seq_len+1) ]

        num_data_per_seq_len = [ max(round(probability * NUM_FULL_BATCH), 1) for probability in prob_of_target_seq_len ]

        input_data_list = []
        
        for target_seq_len, num_data in enumerate(num_data_per_seq_len, 1):
            datas = [DataObject(target_seq_len) for _ in range(num_data)]
            input_data_list.extend(datas)

        num_all_data = DataObject.num_generated_object
        random.shuffle(input_data_list)
        

        ############################################
        start_idx = int(NUM_ENCODER_STEP * 0.9)
        end_idx = int(NUM_ENCODER_STEP * 1)
        
        def is_recording():
            return encoder_step > start_idx and encoder_step <= end_idx
        
        recorded_bsizes = []
        end_batch = []

        # First step of Encoder
        process_dict = {}

        all_size = []

        encoder_step = -1
        for step in range(NUM_ENCODER_STEP * window_size):
            if step != 0 and step % window_size == 0:
                for _ in range(encoder_bsize):
                    data = input_data_list.pop(0)
                    process_dict[data.id] = data
                encoder_step += 1

            if is_recording(): # 특정 Window Index Range 에서의 Running Batch Size 출력
                recorded_bsizes.append(len(process_dict))
            
            #print("current batch : ")
            # Step, 끝난 데이터 개수 출력
            num_local_end_datas = run_process(process_dict)
            end_batch += [num_local_end_datas]
            # print(num_local_end_datas)
            size = len(process_dict)
            all_size.append(size)
        

        # window idx별로 평균 구하기
        window_batch_dic = { key: [] for key in range(window_size) }
        for idx, running_bsize in enumerate(recorded_bsizes):
            window_idx = idx % window_size
            window_batch_dic[window_idx].append(running_bsize)

        bsizes_in_window = [ round(sum(window_batch_dic[window_idx])/len(window_batch_dic[window_idx]), 2) for window_idx in range(window_size) ]
    
        return bsizes_in_window



if __name__ == '__main__':

    ######## PARAMETER ########
    encoder_bsize = 16
    avg_output_seq_len = 32                # Average Sequence Length
    window_size = 24
    ###########################

    result = run_profile_bsize(encoder_bsize, avg_output_seq_len, window_size)
    print(result)