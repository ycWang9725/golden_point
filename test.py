import get_numbers
import sys
import get_numbers
import shutil



# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# 为了“开箱即用”，本脚本没有依赖除了Python库以外的组件。
# 添加自己的代码时，可以自由地引用如numpy这样的组件以方便编程。

import sys
import itertools

from QLearning import *

INPUT_PATH = "Data\FirstData.txt"
LOG_DIR = "logs/"


def LineToNums(line, type=float):
    """将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
    return (type(cell) for cell in line.split('\t'))


def Mean(iter, len):
    """用于计算均值的帮主函数"""
    return sum(iter) / len


def virtual_game():
    if os.path.exists(base_dir + ASSETS):
        shutil.rmtree(base_dir + ASSETS)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(LOG_DIR + "virtual_game.txt", "w+"):
        pass
    with open(INPUT_PATH, "r") as f, open(LOG_DIR + "virtual_game.txt", "a+") as f2:
        sys.stdin = f
        sys.stdout = f2
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

        # q_learn = QLearn(load_path="virtual_game.npy", mode="train", load=False, test=True, bot_num=0,
        #                  rwd_fun="soft", n_steps=5000*(0.8 + 0.1 * (1 - 0)), coding="log-d", state_map="log0.01-50", multi_rwd=True)
        #
        # result = q_learn.next_action(None, None)

        result = get_numbers.get_result([])
        print("%f\t%f" % (result[0], result[1]))
        print(" ")
        for i in range(len(history)):
            if i == 47:
                x = 0
            if i == 150:
                x = 1
            if len(history) == 0:
                result = get_numbers.get_result(history[0: i])
            else:
                result = get_numbers.get_result(history[0: i])
                """
                # 取最近的记录，最多五项。计算这些记录中黄金点的均值作为本脚本的输出。
                candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
                candidate2 = candidate1 * 0.618 # 第二输出。"""

            print("%f\t%f" % (result[0], result[1]))
            print(" ")


def self_game(n_bot=10):
    with open("log.txt", "w+"):
        pass
    with open("log.txt", "a+") as f2:
        sys.stdout = f2

        q_learn = []
        for i in range(n_bot):
            if i < 5:
                q_learn.append(QLearn(mode="train", load=False, load_path="bot_" + str(i) + ".npy", test=True, bot_num=i,
                                      rwd_fun="soft", n_steps=5000*(0.8 + 0.1 * (i - 0)), coding="log-d", state_map="log0.01-50", multi_rwd=True))
            elif i < 10:
                q_learn.append(
                    QLearn(mode="train", load=False, load_path="bot_" + str(i) + ".npy", test=True, bot_num=i,
                           rwd_fun="soft", n_steps=5000 * (0.8 + 0.1 * (i - 5)), coding="log-d",
                           state_map="0-50"))

            # q_learn.append(QLearn(mode="val", load=True, load_path="bot_" + str(i) + ".npy", test=True, bot_num=i, epsilon_th=0.1))
        all_last_actions = []
        output_list = []
        for i in range(n_bot):
            temp = q_learn[i].next_action(None, None)
            all_last_actions.append(temp)
            output_list.append(temp[0])
            output_list.append(temp[1])
        gnum = 0.618 * Mean(output_list, 2*n_bot)
        print(gnum, all_last_actions)

        for _ in range(10000):
            all_last_actions_temp = []
            output_list = []
            for i in range(n_bot):
                temp = q_learn[i].next_action(all_last_actions, gnum)
                all_last_actions_temp.append(temp)
                output_list.append(temp[0])
                output_list.append(temp[1])
            gnum = 0.618 * Mean(output_list, 2 * n_bot)
            all_last_actions = all_last_actions_temp
            print(gnum, all_last_actions)


if __name__ == '__main__':
    #self_game(10)
    virtual_game()

