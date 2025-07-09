import numpy as np
import random

class MutationRLController:
    def __init__(self):
        self.q_table = {}
        self.actions = ['add', 'remove', 'expand', 'fuse', 'none']
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, loss_history, blueprint):
        trend = round(np.mean(np.diff(loss_history[-5:])), 4) if len(loss_history) >= 5 else 0
        complexity = len(blueprint.modules)
        return (trend, complexity)

    def select_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        a_idx = self.actions.index(action)
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][a_idx]
        self.q_table[state][a_idx] += self.lr * td_error
