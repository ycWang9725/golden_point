# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import sys
import os
import shutil
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
import itertools
import numpy as np
from QLearning import *
from iir import *

ASSETS = "/assets"
HISTORY_RANK_DIR = base_dir + ASSETS + "/hist_my_bots_ranks/"


def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))


def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len


def main():

    metaLine = sys.stdin.readline()
    lineNum, columnNum = LineToNums(metaLine, int)

    history = []
    for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
        gnum, *nums = LineToNums(line)
        all_last_actions = []
        for i in range(int(len(nums) / 2)):
            all_last_actions.append([nums[2 * i], nums[2 * i + 1]])
        all_last_actions = np.array(all_last_actions)
        history.append((gnum, all_last_actions))

    if len(history) == 0:
        if os.path.exists(base_dir + ASSETS):
            shutil.rmtree(base_dir + ASSETS)

    result = get_result(history)

    print("%f\t%f" % (result[0], result[1]))


def get_result(history):
    q_learn = QLearn(mode="train", load=True, load_path="bot_" + str(0) + ".npy", test=True, bot_num=None, epsilon_th=0.1, n_steps=50,
                     coding="log-d", state_map="log0.01-50", multi_rwd=True)
    gnums = np.array([history[i][0] for i in range(len(history))])
    if len(gnums) < 2:
        iir_noise_rate = 0.1
    elif len(gnums) < 10:
        iir_noise_rate = np.std(gnums, ddof=1)
    else:
        iir_noise_rate = np.std(gnums[-10:], ddof=1)

    # iir_noise_rate /= 10
    # iir_noise_rate = 0.1

    iir = IIR(alpha=0.5, noise_rate=iir_noise_rate)

    q_i_update_rate = 0.5

    if len(history) == 0:
        result_q = q_learn.next_action(None, None)
        result_i = iir.get_next(None)
        result = result_q
    else:

        last_gnum = q_learn.last_gnum

        if not os.path.exists(HISTORY_RANK_DIR):
            os.makedirs(HISTORY_RANK_DIR)
        if not os.path.exists(HISTORY_RANK_DIR + "q_i.npy"):
            q_i = 0.5
        else:
            q_i = np.load(HISTORY_RANK_DIR + "q_i.npy")
        last_q_dist = np.min(np.abs(q_learn.action2outputs(q_learn.last_action, last_gnum) - history[-1][0]))
        last_i_dist = np.abs(iir.mem - history[-1][0])

        all_last_actions = history[-1][1]
        curr_state = history[-1][0]

        all_last_actions -= curr_state
        all_last_actions = np.abs(all_last_actions)
        bot_action = []

        n_bots = len(all_last_actions)
        # n_bot includes myself in real competition
        for i in range(n_bots):
            bot_action.append(all_last_actions[i][0])
            bot_action.append(all_last_actions[i][1])

        bot_action.sort()

        i_rank = len(bot_action)

        for i in range(len(bot_action)):
            if last_i_dist < bot_action[i]:
                i_rank = i
                break

        q_rank = len(bot_action)
        for i in range(len(bot_action)):
            if last_q_dist < bot_action[i]:
                q_rank = i
                break

        if i_rank < 3 and q_rank >= 3:
            q_i = q_i * (1.0 - q_i_update_rate) + 0.0 * q_i_update_rate
        if q_rank < 3 and i_rank >= 3:
            q_i = q_i * (1.0 - q_i_update_rate) + 1.0 * q_i_update_rate
        if q_rank < 3 and i_rank < 3:
            q_i = q_i * (1.0 - q_i_update_rate) + 0.5 * q_i_update_rate

        # if last_i_dist >= last_q_dist:
        #     q_i = q_i * (1.0 - q_i_update_rate) + 1.0 * q_i_update_rate
        # elif last_q_dist >= last_i_dist:
        #     q_i = q_i * (1.0 - q_i_update_rate) + 0.0 * q_i_update_rate
        # else:
        #     q_i = q_i * (1.0 - q_i_update_rate) + 0.5 * q_i_update_rate

        result_q = q_learn.next_action(history[-1][1], history[-1][0])
        result_i = iir.get_next(history[-1][0])

        random = np.random.rand()
        if random > q_i:
            result = result_i
        else:
            result = result_q

        np.save(HISTORY_RANK_DIR + "q_i.npy", q_i)

        if len(history) == 1:
            result = [history[-1][0] * 0.6, history[-1][0] * 0.5]

    for i in range(np.size(result)):
        if result[i] < 0:
            result[i] = 0.618 * history[-1][0]

    return result


if __name__ == '__main__':
    main()
