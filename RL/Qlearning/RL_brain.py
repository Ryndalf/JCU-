
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


class QlearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QlearningTable, self).__init__()
        print('Initiate Q Learning parameters...')
        for _ in tqdm(range(100)):
            time.sleep(0.01)
        print('\n')
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        rand_flag = True
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
            rand_flag = False
        else:
            action = np.random.choice(self.actions)
        if action == 0:
            print('Choose action: Move up')
        elif action == 1:
            print('Choose action: Move down')
        elif action == 2:
            print('Choose action: Move right')
        else:
            print('Choose action: Move left')
        return action, self.q_table.loc[observation, :], rand_flag

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target-q_predict)
        print(f"Update Q table values at {s} state")


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0]*len(self.actions)