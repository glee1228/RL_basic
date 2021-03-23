import numpy as np

class SmallGridWorld:
    def __init__(self):
        self.grid_size = 4
        self.action = [0, 1, 2, 3]  # up, down, left, right
        self.policy = np.empty([self.grid_size, self.grid_size, len(self.action)], dtype=float)
        self.value = np.zeros((self.grid_size, self.grid_size))
        self.action_reward = -1
        self.discount_factor = 1
        self.policy_evaluation_eps = 10
        def initialize_policy():
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in range(len(self.action)):
                        if i == j and ((i == 0) or (i == self.grid_size-1)):
                            self.policy[i][j] = 0.00
                        else:
                            self.policy[i][j] = 0.25

        initialize_policy()

    def get_state(self, state, action):
        action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        state[0] += action_grid[action][0]
        state[1] += action_grid[action][1]

        state[0] = min(max(state[0], 0),3)
        state[1] = min(max(state[1], 0), 3)

        return state[0], state[1]

    def policy_evaluation(self, max_iter):
        # table initialize
        eps = self.policy_evaluation_eps
        self.policy_evaluation_eps /= 10
        iter = 0
        for iteration in range(max_iter):
            delta = 0
            old_value = self.value.copy()
            for i in range(0, self.value.shape[0]):
                for j in range(0, self.value.shape[1]):
                    old_val = self.value[i][j]
                    if i == j and ((i == 0) or (i == 3)):
                        value_t = 0
                    else:
                        value_t = 0
                        for act in self.action:
                            i_, j_ = self.get_state([i, j], act)
                            value = self.policy[i][j][act] * (self.action_reward + self.discount_factor * old_value[i_][j_])
                            value_t += value

                    self.value[i][j] = round(value_t, 3)
            iter += 1
            print(f'iteration :{iter}')
            print(self.value)
        return self.value

    def run(self):
        iter=100000
        cur_values = self.policy_evaluation(iter)

def save_policy(policy,epoch):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(policy, square=True)
    plt.xlim(0, policy.shape[0])
    plt.ylim(0, policy.shape[1])
    plt.show()
    plt.savefig(f'policy_{epoch}.svg')

if __name__=='__main__':
    sgw = SmallGridWorld()
    sgw.run()