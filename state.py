import numpy as np

class State(object):

    def __init__(self, s):
        self.state = s
        self.left_states = []      # one by one with probability
        self.left_states_prob = []

    def _add(self, left, prob):
        self.left_states.append(left)
        self.left_states_prob.append(prob)


    def _to_numpy(self):
        self.left_states = np.array(self.left_states)
        self.left_states_prob = np.array(self.left_states_prob)

