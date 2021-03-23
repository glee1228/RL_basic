import numpy as np
from tqdm import tqdm
from math import factorial, exp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PoissonDistribution:
    def __init__(self, n, lmbd):
        self.lmbd = lmbd
        self.n = n
        self.vals = {}
        def poisson_dist(n, lamb):
            pd = (lamb ** n) * exp(-lamb) / factorial(n)
            return pd
        pd = [poisson_dist(k, self.lmbd) for k in range(self.n)]
        for k in range(self.n):
            self.vals[k]=pd[k]

class location:
    def __init__(self, req_upper_n, ret_upper_n , req_avg, ret_avg):
        self.requests_avg = req_avg
        self.returns_avg = ret_avg
        self.poisson_req = PoissonDistribution(req_upper_n, self.requests_avg)
        self.poisson_ret = PoissonDistribution(ret_upper_n, self.returns_avg)

class JacksCarRental:
    def __init__(self):
        self.max_num_cars = 20
        self.discount_factor = 0.9
        self.rent_reward = 10
        self.action_reward = -2
        self.A = location(7, 7, 3, 3)
        self.B = location(8, 6, 4, 2)
        self.value = np.zeros((self.max_num_cars + 1, self.max_num_cars + 1))
        self.policy = np.zeros((self.max_num_cars + 1, self.max_num_cars + 1), dtype=int)
        self.policy_evaluation_eps = 50

    def expected_reward(self, state, action):
        reward = 0.0
        new_state = [max(min(state[0] - action, self.max_num_cars), 0), max(min(state[1] + action, self.max_num_cars), 0)]

        reward = reward + self.action_reward * abs(action)

        for A_req in range(0, self.A.poisson_req.n):
            for B_req in range(0, self.B.poisson_req.n):
                for A_ret in range(0, self.A.poisson_ret.n):
                    for B_ret in range(0, self.B.poisson_ret.n):
                        event_prob = self.A.poisson_req.vals[A_req] * self.B.poisson_req.vals[B_req] * \
                                     self.A.poisson_ret.vals[A_ret] * self.B.poisson_ret.vals[B_ret]

                        valid_req_A = min(new_state[0], A_req) # if request > return -> return
                        valid_req_B = min(new_state[1], B_req)

                        rew = (valid_req_A + valid_req_B) * (self.rent_reward)

                        new_s = [0, 0]
                        new_s[0] = max(min(new_state[0] - valid_req_A + A_req, self.max_num_cars), 0)
                        new_s[1] = max(min(new_state[1] - valid_req_B + B_ret, self.max_num_cars), 0)

                        # E[R_{t+1}+\gamma value(S_{t+1})|S_t=s] = E[R_{t+1}+\gammaR_{t+2}+...|S_t=s] by iterative expectation
                        reward += event_prob * (rew + self.discount_factor * self.value[new_s[0]][new_s[1]])

        return reward

    def policy_evaluation(self, epoch):
        bar = tqdm(range(1), ncols=300)
        deltas = AverageMeter()
        values = AverageMeter()
        eps = self.policy_evaluation_eps

        self.policy_evaluation_eps /= 10

        while True:
            delta = 0
            for i in range(self.value.shape[0]):
                for j in range(self.value.shape[1]):
                    old_value = self.value[i][j]
                    self.value[i][j] = self.expected_reward([i, j], self.policy[i][j])
                    delta = max(delta, abs(self.value[i][j] - old_value))
                    deltas.update(delta)

            values.update(np.mean(self.value))
            bar.set_description(f'[Epoch {epoch}]-policy_evaluation delta/eps: {delta:.2f}({deltas.avg:.2f})/{eps:.2f}, '
                                f'value: {np.mean(self.value):.4f}({values.avg:.4f})  ')
            bar.update()
            if delta < eps:
                bar.close()
                break

    def policy_improvement(self, epoch):
        old_policy = self.policy.copy()
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                max_act_val = None
                max_act = None

                tau_12 = min(i, 5)
                tau_21 = -min(j, 5)

                for act in range(tau_21, tau_12 + 1):
                    sigma = self.expected_reward([i, j], act)
                    if max_act_val == None:
                        max_act_val = sigma
                        max_act = act
                    elif max_act_val < sigma:
                        max_act_val = sigma
                        max_act = act

                self.policy[i][j] = max_act
        print(f'\n[Epoch {epoch}]-policy_improvement policy:\n {self.policy} ')

        return old_policy, self.policy

    def run(self):
        epoch = 0
        while True:
            self.policy_evaluation(epoch)
            old_pi, cur_pi = self.policy_improvement(epoch)
            policy_changes = np.sum(old_pi != cur_pi)
            print(f'policy_changes : {policy_changes}')
            save_policy(cur_pi,epoch)
            # print()
            # print(old_pi)
            # print()
            # print(cur_pi)
            epoch += 1
            if policy_changes == 0:
                break


def save_policy(policy,epoch):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(policy, square=True)
    plt.xlim(0, policy.shape[0])
    plt.ylim(0, policy.shape[1])
    plt.show()
    plt.savefig(f'policy_{epoch}.svg')


if __name__=='__main__':
    jcr = JacksCarRental()
    jcr.run()