import numpy as np
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

ASSETS = "/assets"


Q_TABLE_DIR = base_dir + ASSETS + "/q_table/"
LOG_DIR = base_dir + ASSETS + "/logs/"
EPSILON_DIR = base_dir + ASSETS + "/epsilon/"
LAST_GNUM_DIR = base_dir + ASSETS + "/last_gnum/"
LAST_STATE_DIR = base_dir + ASSETS + "/last_state/"
LAST_ACTION_DIR = base_dir + ASSETS + "/last_action/"


class QLearn:
    def __init__(self, load=False, load_path=None, test=False, bot_num=None, n_steps=5000, epsilon_th=0, mode="train",
                 rwd_fun="soft", state_map="0-50", coding="uniform", multi_rwd=False):
        self.load_path = load_path
        self.mode = mode
        self.multi_rwd = multi_rwd
        if not os.path.exists(Q_TABLE_DIR):
            os.makedirs(Q_TABLE_DIR)
        if not os.path.exists(EPSILON_DIR):
            os.makedirs(EPSILON_DIR)
        if not os.path.exists(LAST_GNUM_DIR):
            os.makedirs(LAST_GNUM_DIR)
        if not os.path.exists(LAST_STATE_DIR):
            os.makedirs(LAST_STATE_DIR)
        if not os.path.exists(LAST_ACTION_DIR):
            os.makedirs(LAST_ACTION_DIR)
        if load:
            if not os.path.exists(Q_TABLE_DIR + self.load_path):
                self.q_table = [np.zeros((100, 100), dtype=np.float32),
                                np.zeros((100, 100), dtype=np.float32)]
            else:
                self.q_table = np.load(Q_TABLE_DIR + self.load_path)

            if not os.path.exists(EPSILON_DIR + self.load_path):
                self.epsilon = 1
            else:
                self.epsilon = np.load(EPSILON_DIR + self.load_path)

            if not os.path.exists(LAST_GNUM_DIR + self.load_path):
                self.last_gnum = 19.5
            else:
                self.last_gnum = np.load(LAST_GNUM_DIR + self.load_path)

            if not os.path.exists(LAST_STATE_DIR + self.load_path):
                self.last_state = self.prob2action(np.max(self.q_table[0], axis=1))
            else:
                self.last_state = np.load(LAST_STATE_DIR + self.load_path)

            if not os.path.exists(LAST_ACTION_DIR + self.load_path):
                self.last_action = np.array([self.prob2action(self.q_table[0][self.last_state]),
                                             self.prob2action(self.q_table[1][self.last_state])])
            else:
                self.last_action = np.load(LAST_ACTION_DIR + self.load_path)

        else:
            self.q_table = [np.zeros((100, 100), dtype=np.float32),
                            np.zeros((100, 100), dtype=np.float32)]
            self.epsilon = 1
            self.last_state = int(8)
            self.last_gnum = 19.5
            self.last_action = np.array([19, 31], dtype=np.int)
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon_th = epsilon_th
        self.epsilon_decay = 1 - 1/n_steps
        self.test = test
        self.bot_num = bot_num
        self.rwd_fun = rwd_fun
        self.state_map = state_map
        self.coding = coding
        if self.multi_rwd:
            self.epsilon = 0
        self.action_alpha = np.exp(np.log(3/0.3)/100)
        self.action_beta = -np.log(0.3)/np.log(self.action_alpha)
        if self.state_map == "log0.01-50":
            self.state_alpha = np.log(50 / 0.01) / 100
        elif self.state_map == "log0.1-50":
            self.state_alpha = np.log(50 / 0.1) / 100

        if self.test:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            if self.bot_num is not None:
                with open(LOG_DIR + "bot_" + str(self.bot_num) + ".txt", "w+"):
                    pass
                with open(LOG_DIR + "bot_" + str(self.bot_num) + ".txt", "a+") as f:
                    old_out = sys.stdout
                    sys.stdout = f
                    print("epsilon_decay: " + str(self.epsilon_decay))
                    sys.stdout = old_out

    def update_epsilon(self):
        if self.mode == "train":
            self.epsilon *= self.epsilon_decay
        else:
            assert self.mode == "val"
            self.epsilon = self.epsilon_th
        np.save(EPSILON_DIR + self.load_path, self.epsilon)

    def next_action(self, all_last_actions, curr_state):
        if curr_state is not None:
            self.update_q_table(all_last_actions, curr_state)

            rr = np.random.rand(2)
            rs = np.random.rand(2)
            action = []
            random_state_0 = int(rs[0] * 100)  # Random select
            random_state_1 = int(rs[1] * 100)  # Random select

            self.last_gnum = curr_state
            curr_state = self.gnum2state(curr_state)

            if rr[0] < self.epsilon:
                action.append(random_state_0)
            else:
                action.append(self.prob2action(self.q_table[0][curr_state]))

            if rr[1] < self.epsilon:
                action.append(random_state_1)
            else:
                action.append(self.prob2action(self.q_table[1][curr_state]))

            action = np.array(action)

            self.last_action = action
            self.last_state = curr_state
        else:
            self.last_gnum = 19.5

            self.update_epsilon()

        np.save(LAST_ACTION_DIR + self.load_path, self.last_action)
        np.save(LAST_GNUM_DIR + self.load_path, self.last_gnum)
        np.save(LAST_STATE_DIR + self.load_path, self.last_state)

        return self.action2outputs(self.last_action, self.last_gnum)

    def prob2action(self, prob):
        if self.multi_rwd:
            return self.random_softmax(prob)
        else:
            return np.argmax(prob)

    def action2outputs(self, last_action, last_gnum):
        if self.coding == "uniform":
            return last_action + 0.5
        else:
            assert self.coding == "log-d"
            out = last_gnum * self.action_alpha ** (last_action - self.action_beta)
            for i in range(np.size(last_action)):
                if out[i] < 0.1 ** 9:
                    out[i] = 0.1 ** 9
            return out

    def update_q_table(self, all_last_actions, curr_state):
        if self.multi_rwd is False:
            reward = self.calculate_reward(all_last_actions, curr_state)
            alpha = self.alpha
            gamma = self.gamma
            last_state = self.last_state
            last_action = self.last_action
            curr_state = self.gnum2state(curr_state)
            if self.mode != "train":
                return
            for i in range(2):
                try:
                    self.q_table[i][last_state][last_action[i]] = (1-alpha) * self.q_table[i][last_state][last_action[i]] + \
                    alpha * (reward + gamma * np.max(self.q_table[i][curr_state]))
                except:
                    a = 1
            np.save(Q_TABLE_DIR + self.load_path, self.q_table)
        else:
            reward = self.calculate_multi_reward(all_last_actions, curr_state)
            alpha = self.alpha
            gamma = self.gamma
            last_state = self.last_state
            last_action = self.last_action
            curr_state = self.gnum2state(curr_state)
            if self.mode != "train":
                return
            for i in range(2):
                try:
                    self.q_table[i][last_state] = (1 - alpha) * self.q_table[i][last_state] + \
                                                  alpha * (reward + gamma * np.max(self.q_table[i][curr_state]))
                except:
                    a = 1
            np.save(Q_TABLE_DIR + self.load_path, self.q_table)

    def calculate_reward(self, all_last_actions, curr_state):
        all_last_actions -= curr_state
        all_last_actions = np.abs(all_last_actions)
        last_action = np.abs(self.last_action + 0.5 - curr_state)
        bot_action = []
        if self.bot_num is None:
            bot_action.append((0, last_action[0]))
            bot_action.append((0, last_action[1]))
            n_bots = len(all_last_actions) + 1
            # n_bot dose not includes myself in virtual competition
            for i in range(n_bots - 1):
                bot_action.append((i + 1, all_last_actions[i][0]))
                bot_action.append((i + 1, all_last_actions[i][1]))
        else:
            n_bots = len(all_last_actions)
            # n_bot includes myself in real competition
            for i in range(n_bots):
                bot_action.append((i, all_last_actions[i][0]))
                bot_action.append((i, all_last_actions[i][1]))

        bot_action.sort(key=lambda x: x[1])
        bot_rank = []
        for i in range(len(bot_action)):
            bot_rank.append((bot_action[i][0], i))

        bot_rank.sort(key=lambda x: x[0])

        if self.bot_num is None:
            my_rank = np.array([bot_rank[0][1], bot_rank[1][1]])
        else:
            my_rank = np.array([bot_rank[2*self.bot_num][1], bot_rank[2*self.bot_num+1][1]])

        rewards = self.my_rank2score(my_rank, n_bots)

        if self.test:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            if self.bot_num is not None:
                with open(LOG_DIR + "bot_" + str(self.bot_num) + ".txt", "a+") as f:
                    old_out = sys.stdout
                    sys.stdout = f
                    print("Output1: " + str(self.last_action[0] + 0.5) + " Output2: " + str(self.last_action[1] + 0.5))
                    print("Rewards: " + str(rewards[0]) + " " + str(rewards[1]))
                    print("My_rank: " + str(my_rank[0]) + " " + str(my_rank[1]))
                    print("Current: " + str(curr_state))
                    sys.stdout = old_out
            else:
                with open(LOG_DIR + "virtual_game.txt", "a+") as f:
                    old_out = sys.stdout
                    sys.stdout = f
                    print("Rewards: " + str(rewards[0]) + " " + str(rewards[1]))
                    print("My_rank: " + str(my_rank[0]) + " " + str(my_rank[1]))
                    print("Current: " + str(curr_state))
                    sys.stdout = old_out

        return np.sum(rewards)

    def rank2score(self, rank, n_bots):
        if self.rwd_fun == "soft":
            return (n_bots - (n_bots + 2) * rank / 2 / (n_bots - 1)) / (n_bots + 2)
        # elif self.rwd_fun == "hard":
        #     if rank == 0:
        #         return n_bots / (n_bots + 2)
        #     elif rank == 2 * n_bots - 1:
        #         return -2 / (n_bots + 2)
        #     else:
        #         return 0
        elif self.rwd_fun == "norm_soft":
            return (n_bots - (n_bots + 2) * rank / (n_bots - 1)) / (n_bots + 2)
        else:
            assert self.rwd_fun == "avg"
            result = 0
            if rank == 0:
                result = n_bots
            elif rank == 2 * n_bots - 1:
                result = -2
            result /= (n_bots + 2)
            result += (n_bots - (n_bots + 2) * rank / 2 / (n_bots - 1)) / (n_bots + 2)
            return result


    def my_rank2score(self, my_rank, n_bots):
        return np.array([self.rank2score(my_rank[i], n_bots) for i in range(len(my_rank))])

    def gnum2state(self, gnum):
        if self.state_map == "0-50":
            result = int(gnum * 2)
            if result > 99:
                result = 99
            return result
        elif self.state_map == "log0.01-50":
            result = np.log(gnum/0.01)/self.state_alpha
            if result > 99:
                result = 99
            elif result < 0:
                result = 0
            return int(result)
        else:
            assert self.state_map == "log0.1-50"
            result = np.log(gnum/0.1)/self.state_alpha
            if result > 99:
                result = 99
            elif result < 0:
                result = 0
            return int(result)

    def calculate_multi_reward(self, all_last_actions, curr_state):
        all_last_actions -= curr_state
        all_last_actions = np.abs(all_last_actions)
        last_action = np.abs(self.last_action + 0.5 - curr_state)
        bot_action = []

        n_bots = len(all_last_actions)
        # n_bot includes myself in real competition
        for i in range(n_bots):
            bot_action.append(all_last_actions[i][0])
            bot_action.append(all_last_actions[i][1])

        bot_action.sort()

        all_posible_last_outputs = self.action2outputs(np.linspace(0, 99, 100, True, dtype=int), self.last_gnum)
        all_posible_last_distance = np.abs(all_posible_last_outputs - curr_state)
        all_posible_last_distance = [(i, all_posible_last_distance[i]) for i in range(np.size(all_posible_last_distance))]
        all_posible_last_distance.sort(key=lambda x: x[1])

        all_posible_rank_action = []
        p_bot_action = 0
        end_flag = 0
        len_bot_action = len(bot_action)
        for p_all_distance in range(len(all_posible_last_distance)):
            try:
                if all_posible_last_distance[p_all_distance][1] < bot_action[p_bot_action]:
                    all_posible_rank_action.append((p_bot_action, all_posible_last_distance[p_all_distance][0]))
                else:
                    while all_posible_last_distance[p_all_distance][1] >= bot_action[p_bot_action] and end_flag == 0:
                        if p_bot_action < len_bot_action - 1:
                            p_bot_action += 1
                        else:
                            end_flag = 1
                    all_posible_rank_action.append((p_bot_action, all_posible_last_distance[p_all_distance][0]))
            except:
                a = 1
        all_posible_rank_action.sort(key=lambda x: x[1])
        all_posible_ranks = [all_posible_rank_action[i][0] for i in range(len(all_posible_rank_action))]

        rewards = self.my_rank2score(all_posible_ranks, n_bots)

        return rewards

    def random_softmax(self, vector):
        mean = np.mean(vector)
        vector = vector - mean
        vector = np.exp(vector * 150)
        sum_exp = np.sum(vector)
        vector = vector/sum_exp
        accumulate_vector = 0
        random = np.random.rand()
        for i in range(np.size(vector)):
            accumulate_vector += vector[i]
            if random < accumulate_vector:
                return i
        return np.size(vector) - 1