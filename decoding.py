from generate_matrix import generate_A_matrix, read
import numpy as np
from collections import deque
import time


def read_ppg(filename, threshold):
    # arrays = []
    with open(filename, 'r') as f:
        frame = 0
        post_array = []
        for i, line in enumerate(f):
            if "[" in line:
                print("current audio is:{}".format(frame))
                # arrays.append(post_array)
                yield post_array
                post_array = []
                frame += 1
            else:
                cur = line.split(' ')[2:-1]
                post = np.array([float(r) for r in cur])
                post_array.append(post)
        # arrays.append(post_array)
        yield post_array
    # return np.array(arrays)

def read(file='./map'):
    sp_dict = {}
    with open(file, 'r') as f:
        for item in f:
            ind, *phone = item.split()
            sp_dict[int(ind)] = phone

    return sp_dict

# def decode(filename, threshold=0.0):
#     for audio_array in read_ppg(filename, threshold):
#         if audio_array:
#             initial_state = 0
#             dp_matrix = np.array(audio_array)
#             best_pre = np.ndarray(dp_matrix.shape, dtype=int)
#
#             # compute initial condition, T=1:
#             for i in range(M):
#                 ob = Obs[0]
#                 prob_hs[0][i] = X[i] * B[i][ob]
#                 best_pre[0][i] = 0

def decode(audio_array, A_matrix, threshold=0.0):
    T, M = len(audio_array), len(audio_array[0])
    dp_matrix = np.zeros((T, M))
    best_pre = np.ndarray(dp_matrix.shape, dtype=int)
    # compute initial condition, T=1:
    # initial_state = np.full(M, 1/M)
    # for i in range(M):
    #     dp_matrix[0][i] = initial_state[i] * audio_array[0][i]
    #     best_pre[]

    # t=1
    # dp_matrix[0] = audio_array[0]
    # best_pre[0] = -1
    #
    # # iterate t from 2 to T
    # for t in range(1, T):
    #     for i in range(M):
    #         # possible_next, probabilities = zip(*A_matrix[i])
    #         dp_matrix[t][i], best_pre[t][i] = max((dp_matrix[t - 1][j] * v, j) for j, v in A_matrix[i])
    #         dp_matrix[t][i] *= audio_array[t][i]
    #
    # # compute the maximum probability of time T
    # max_i = np.argmax(dp_matrix[-1])
    # max_v = dp_matrix[-1][max_i]
    # best_seq = deque([max_i])
    #
    # # backtracking to get the best sequence
    # for t in range(T - 2, -1, -1):
    #     max_i = best_pre[t + 1][max_i]
    #     best_seq.appendleft(max_i)

    # t=1
    initial_state = 0
    dp_matrix[0][initial_state] = 1 * audio_array[0][initial_state]
    best_pre[0] = -1

    # iterate t from 1 to T-1, every time we update t+1 's probability
    for t in range(T - 1):
        possible_node = dp_matrix[t].argsort()[-10:]
        for i in possible_node:
            if dp_matrix[t][i] > 0:
                # possible_next, probabilities = zip(*A_matrix[i])
                for (j, v) in A_matrix[i]:
                    cur_trans_prob = dp_matrix[t][i] * v
                    if dp_matrix[t + 1][j] < cur_trans_prob:
                        dp_matrix[t + 1][j] = cur_trans_prob
                        best_pre[t + 1][j] = i
        dp_matrix[t + 1] *= audio_array[t + 1]

    # compute the maximum probability of time T
    # sp_dict = read()
    max_i = np.argmax(dp_matrix[-1])
    max_v = dp_matrix[-1][max_i]
    # best_seq = deque([sp_dict[max_i]])
    best_seq = deque([max_i])

    # backtracking to get the best sequence
    for t in range(T - 2, -1, -1):
        max_i = best_pre[t + 1][max_i]
        best_seq.appendleft(max_i)

    return best_seq, max_v


def decode_top_k(audio_array, A_matrix, threshold=0.0, top_k=5):
    T, M = len(audio_array), len(audio_array[0])
    dp_matrix = np.zeros((T, M))
    best_pre = np.ndarray(dp_matrix.shape, dtype=int)
    initial_state = 0
    dp_matrix[0][initial_state] = 1 * audio_array[0][initial_state]
    best_pre[0] = -1

    # iterate t from 1 to T-1, every time we update t+1 's probability
    for t in range(T - 1):
        for i in range(M):
            if dp_matrix[t][i] > 0:
                for (j, v) in A_matrix[i]:
                    cur_trans_prob = dp_matrix[t][i] * v
                    if dp_matrix[t + 1][j] < cur_trans_prob:
                        dp_matrix[t + 1][j] = cur_trans_prob
                        best_pre[t + 1][j] = i

        dp_matrix[t + 1] *= audio_array[t + 1]

    # compute the maximum probability of time T
    # max_i = np.argmax(dp_matrix[-1])

    top_k_index = dp_matrix[-1].argsort()[-top_k:][::-1]

    top_k_v = [dp_matrix[-1][i] for i in top_k_index]

    top_k_path = []

    sp_dict = read()

    for max_i in top_k_index:
        best_seq = deque([sp_dict[max_i]])

        # backtracking to get the best sequence
        for t in range(T - 2, -1, -1):
            max_i = best_pre[t + 1][max_i]
            best_seq.appendleft(sp_dict[max_i])

        top_k_path.append(best_seq)

    return top_k_path, top_k_v



# def decode_audios(filename, threshold=0.0):
#     A = generate_A_matrix()
#     for audio_array in read_ppg(filename, threshold):
#         if audio_array:
#             best_seq, max_v = decode(audio_array, A)
#             with open('out.txt', 'w') as f:
#                 f.write(str(best_seq))
#             print(best_seq)
#             print(max_v)
#             assert 1 == 2



def viterbi(audio_array, A_matrix, threshold=0.0):
    T, M = len(audio_array), len(audio_array[0])
    dp_matrix = np.array(audio_array)
    best_pre = np.ndarray(dp_matrix.shape, dtype=int)
    # compute initial condition, T=1:
    best_pre[0] = -1

    # iterate t from 2 to T
    for t in range(1, T):
        for i in A_matrix:
            vec = A_matrix[i].left_states_prob * np.array([dp_matrix[t-1][j] for j in A_matrix[i].left_states])
            dp_matrix[t][i] = vec.max()
    # , best_pre[t][i]
    # # compute the maximum probability of time T
    max_i = np.argmax(dp_matrix[-1])
    max_v = dp_matrix[-1][max_i]
    # best_seq = deque([max_i])

    # backtracking to get the best sequence
    # for t in range(T - 2, -1, -1):
    #     max_i = best_pre[t + 1][max_i]
    #     best_seq.appendleft(max_i)
    #
    # return max_v, best_seq


def decode_audios(filename, out_file, threshold=0.0):
    A = generate_A_matrix()
    with open(out_file, 'w') as f:
        for i, audio_array in enumerate(read_ppg(filename, threshold)):
            if audio_array:
                print('Start decoding audio:{}'.format(i))
                seq, v = decode(audio_array, A)
                # write the result to txt file
                f.write('Audio{} best decoding seq:\n'.format(i))
                f.write(str(seq) + '\n\n')
                print(v)

def main():
    decode_audios('../output_offline', 'out_full_ids.txt')
    # decode_audios('/home/maybe/TAGyou-Lite/ASR/engine/xiaoqing_maybe/ppg_maybe', 'ppg_maybe_top5_out.txt')
    # decode_audios('/home/maybe/TAGyou-Lite/ASR/engine/xiaoqing_maybe/ppg_neg', 'ppg_neg_top5_out.txt')
    # A = generate_A_matrix()
    # array = read_ppg('./output_offline', 0.0)[1]
    # s = time.time()
    # print('Start decoding')
    # best, v = decode(array, A)
    # # best_top, vs = decode_top_k(array, A)
    # print('Cost: {}'.format(time.time() - s))
    # print(best)
    # # print(best_top[0])
    # print(v)
    # print(vs)

if __name__ == '__main__':
    main()
