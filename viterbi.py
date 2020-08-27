import numpy as np
from collections import deque

# six elements of HMM


H = ['box1', 'box2', 'box3']
O = ['red', 'white']

A_matrix = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B_matrix = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
X_vector = [0.2, 0.4, 0.4]
obs_vector = [0, 1, 0]




def viterbi(H, O, A, B, X, Obs):
    '''
    :param H:  set of hidden states, length of M
    :param O:  set of observation states, length of N
    :param A:  hidden states transition probability matrix, shape of (M, M)
    :param B:  observed states happened probability matrix, shape of (M, N)
    :param X:  initial probability of hidden states
    :param Obs: Observation sequence,length of T

    :return: maximum probability sequence of hidden states that computed by Viterbi algorithm
    '''
    M, N, T = len(H), len(O), len(Obs)
    prob_hs = np.ndarray((T, M))
    best_pre = np.ndarray((T, M), dtype=int)

    # compute initial condition, T=1:
    for i in range(M):
        ob = Obs[0]
        prob_hs[0][i] = X[i] * B[i][ob]
        best_pre[0][i] = 0

    # iterate form time 2 to T
    for t in range(1, T):
        ob = Obs[t]
        for i in range(M):
            prob_hs[t][i], best_pre[t][i] = max((prob_hs[t - 1][j] * A[j][i], j) for j in range(M))
            prob_hs[t][i] *= B[i][ob]

    # compute the maximum probability of time T
    max_i = np.argmax(prob_hs[-1])
    max_v = prob_hs[-1][max_i]
    best_seq = deque([H[max_i]])

    # backtracking to get the best sequence
    for t in range(T - 2, -1, -1):
        max_i = best_pre[t + 1][max_i]
        best_seq.appendleft(H[max_i])

    return best_seq, max_v


if __name__ == '__main__':
    print(viterbi(H, O, A_matrix, B_matrix, X_vector, obs_vector))
