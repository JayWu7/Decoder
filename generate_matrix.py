import numpy as np
from state import State


def read_data(filename):
    forward_dict = {}
    self_loop_dict = {}
    phone_dict = {}
    left_dict = {}
    forward_phone_dict = {}
    self_loop_phone_dict = {}
    # drop_set = {1437, 10385, 9652}

    with open(filename, 'r') as f:
        for line in f:
            cur = line.split(' ')
            left = int(cur[0])
            pho = int(cur[1])
            forward = int(cur[2])
            self_loop = int(cur[3])

            if left not in left_dict:
                left_dict[left] = {forward}
            else:
                left_dict[left].add(forward)

            if pho not in phone_dict:
                phone_dict[pho] = {forward}
            else:
                phone_dict[pho].add(forward)

            if forward not in forward_dict:
                forward_dict[forward] = {self_loop}
                forward_phone_dict[forward] = {pho}
            else:
                forward_dict[forward].add(self_loop)
                forward_phone_dict[forward].add(pho)

            if self_loop not in self_loop_dict:
                self_loop_dict[self_loop] = {forward}
                self_loop_phone_dict[self_loop] = {pho}
            else:
                self_loop_dict[self_loop].add(forward)
                self_loop_phone_dict[self_loop].add(pho)

    # [(forward_dict.pop(d), self_loop_dict.pop(d)) for d in drop_set]  # delete unrelated item

    return left_dict, phone_dict, forward_dict, self_loop_dict, forward_phone_dict, self_loop_phone_dict


# def generate_A_matrix(dimension=10456):
#     '''
#     :param dimension: dimensions of HMM ids
#     :return: transition matrix of HMM
#     '''
#     forward_dict, self_loop_dict = read_data('./cdphones.int')
#     matrix = np.zeros((dimension, dimension))
#
#     for forward in forward_dict:
#         possible_set = set(forward_dict.keys()) - {forward} | forward_dict[forward]
#         aver_prob = 1 / len(possible_set)
#         for item in possible_set:
#             matrix[forward][item] = aver_prob
#
#     for self_loop in self_loop_dict:
#         matrix[self_loop][self_loop] = 0.5
#         possible_set = set(forward_dict.keys()) - self_loop_dict[self_loop]
#         aver_prob = 0.5 / len(possible_set)
#         for item in possible_set:
#             matrix[self_loop][item] = aver_prob
#
#     return matrix


def generate_A_matrix():
    '''
    :param dimension: dimensions of HMM ids
    :return: transition matrix of HMM
    '''
    matrix = {}
    left_dict, photo_dict, forward_dict, self_loop_dict, forward_phone_dict, self_loop_phone_dict = read_data(
        './cdphones.int')

    for forward in forward_dict:
        phones = forward_phone_dict[forward]
        possible_set = set()
        for phone in phones:
            possible_set |= left_dict[phone]

        possible_set_af = possible_set - {forward}
        possible_set_sf = forward_dict[forward]

        aver_prob_af = 0.5 / len(possible_set_af)
        aver_prob_sf = 0.5 / len(possible_set_sf)

        next_state_distribution = [(state, aver_prob_af) for state in possible_set_af] + [(state, aver_prob_sf) for
                                                                                          state in possible_set_sf]
        matrix[forward] = next_state_distribution

    for self_loop in self_loop_dict:
        phones = self_loop_phone_dict[self_loop]
        possible_set = set()
        for phone in phones:
            possible_set |= left_dict[phone]
        possible_set = possible_set - self_loop_dict[self_loop]
        aver_prob = 0.5 / len(possible_set)

        next_state_distribution = [(self_loop, 0.5)] + [(state, aver_prob) for state in possible_set]
        matrix[self_loop] = next_state_distribution

    return matrix


# def generate_A_new(A_matrix):
#     matrix = generate_A_matrix_full(A_matrix)
#
#     for i, row in enumerate(matrix):
#         for j,v in enumerate(row):
#             if v > 0:


    max_l = 0
    max_i = -1
    for key, state in matrix.items():
        state._to_numpy()
        if len(state.left_states_prob) > max_l:
            max_l = len(state.left_states_prob)
            max_i = key

    print(max_l)
    print(max_i)
    return matrix

def generate_A_matrix_full(A_matrix):
    full_A = np.zeros((10456, 10456))
    for k, values in A_matrix.items():
        for i,v in values:
            full_A[k][i] = v
    return full_A.T


def read(file='./map'):
    sp_dict = {}
    with open(file, 'r') as f:
        for item in f:
            ind, *phone = item.split()
            sp_dict[int(ind)] = phone


    return sp_dict


matrix = generate_A_matrix_full(generate_A_matrix())
# print(matrix[0].left_states_prob)
# print(matrix[16].left_states)
# print(len(matrix[16].left_states))
# print(len(matrix))
# print(type(matrix))
# print(matrix[0])
# print(len(matrix[0]))
# for key,value in matrix.items():
#     print(111111)
#     print(key)
#     print(len(value))
