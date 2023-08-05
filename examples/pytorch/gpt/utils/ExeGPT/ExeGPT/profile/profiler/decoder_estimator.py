from math import ceil, floor, factorial
from scipy.stats import poisson

# def estimate_encoder_batch(lamb, decoder_batch_size, max_seq, window_size) -> list:
#     # decoder_batch_size = ceil(decoder_batch_size / 6)
#     probability_list = [0. for _ in range(window_size)]
#     for l in range(1, max_seq + 1):
#         distribution = pois_dist(l, lamb, max_seq)
#         for e in range(1, window_size + 1):
#             if l % window_size == e or (l % window_size == 0 and window_size == e):
#                 probability_list[e - 1] += 1 / ceil(l / window_size) * distribution
#                 break

#     batch_list = [decoder_batch_size]

#     acc_output_p = 0
#     threshold = 1. / decoder_batch_size
#     for idx, p in enumerate(probability_list):
#         if acc_output_p != 0:
#             p += acc_output_p

#         output_batch = floor(decoder_batch_size * p)

#         if output_batch == 0:
#             acc_output_p = p
#         else:
#             acc_output_p = p - threshold * output_batch

#         batch_list.append(batch_list[idx] - output_batch)

#     embd_batch_list = []
#     last_batch = batch_list[0]
#     for n in batch_list[1:]:
#         embd_batch_list.append(last_batch-n)
#         last_batch = n
    
#     batch_list[-1] -= round(acc_output_p)
#     encoder_batch = decoder_batch_size - batch_list[-1]
    
#     return batch_list[:-1], encoder_batch, embd_batch_list * 2


def estimate_decoder_batch(lamb, encoder_batch_size, max_seq, window_size) -> list:
    # decoder_batch_size = ceil(decoder_batch_size / 6)
    probability_list = [0. for _ in range(window_size)]
    for l in range(1, max_seq + 1):
        distribution = pois_dist(l, lamb, max_seq)
        for e in range(1, window_size + 1):
            if l % window_size == e or (l % window_size == 0 and window_size == e):
                probability_list[e - 1] += 1 / ceil(l / window_size) * distribution
                break
            
    decoder_batch_size = encoder_batch_size // sum(probability_list)
    
    batch_list = [decoder_batch_size]

    acc_output_p = 0
    threshold = 1. / decoder_batch_size
    for idx, p in enumerate(probability_list):
        if acc_output_p != 0:
            p += acc_output_p

        output_batch = floor(decoder_batch_size * p)

        if output_batch == 0:
            acc_output_p = p
        else:
            acc_output_p = p - threshold * output_batch

        batch_list.append(batch_list[idx] - output_batch)

    embd_batch_list = []
    last_batch = batch_list[0]
    for n in batch_list[1:]:
        embd_batch_list.append(last_batch-n)
        last_batch = n
    
    return batch_list[:-1], embd_batch_list


def pois_dist(n, lamb, max_seq):
    assert n <= max_seq, "n > max_seq"
    
    if max_seq == n:
        return 1 - poisson.cdf(max_seq, mu=lamb)
    else:
        return poisson.pmf(n, mu=lamb)
    
    
def uniform_dist(n, lamb, max_seq):
    # assert n <= max_seq, "n > max_seq"
    return 1 / max_seq

if __name__ == "__main__":
    _LAMBDA = 128
    batch_list = [21]
    for batch in batch_list:
        print(estimate_decoder_batch(_LAMBDA, batch, _LAMBDA*2, 128))
    
    # print(pois_dist(_MAX_SEQ, _LAMBDA, _MAX_SEQ))
    # print(1 - poisson.cdf(_MAX_SEQ, mu=_LAMBDA))
    
    # print(f"n: {_MAX_SEQ} | {pois_dist(_MAX_SEQ, _LAMBDA, _MAX_SEQ)}")

    # a = 0.
    # for n in range(1, _MAX_SEQ):
    #     a += pois_dist(n, _LAMBDA, _MAX_SEQ)
    # print(f"totla: {1 - a}")
    
    
    
    
    
    #     assert n <= max_seq, "n > max_seq"
    
    # if max_seq == n:
    #     return 1 - poisson.cdf(max_seq, mu=lamb)
    # else:
    #     return poisson.pmf(n, mu=lamb)