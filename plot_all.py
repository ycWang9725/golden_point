import matplotlib.pyplot as plt
import numpy as np
import re
import os

LOG_DIR = "logs/"
SAVE_FIG_DIR = "fig/"

def plot_all(n_bot, mode="train"):
    th = []
    for i in range(n_bot):
        th_temp, mean_reward_temp, mean_hd_rwd_temp = plot_one(i, n_bot)
        th.append((th_temp, mean_reward_temp, mean_hd_rwd_temp))
    if mode == "train":
        th.sort(key=lambda x: x[0])
        th_plt = np.array([th[i][0] for i in range(len(th))])
    else:
        th_plt = np.linspace(1/2/n_bot, 1-1/2/n_bot, n_bot, True, dtype=np.float)
    mean_reward_plt = np.array([th[i][1] for i in range(len(th))])
    mean_hd_rwd_plt = np.array([th[i][2] for i in range(len(th))])

    if not os.path.exists(SAVE_FIG_DIR):
        os.makedirs(SAVE_FIG_DIR)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(th_plt, mean_reward_plt, lw=1, color='green', linestyle="-")
    title = "Threshold-mean_reward"
    plt.title(title)
    plt.ion()
    plt.savefig(SAVE_FIG_DIR + title.replace(" ", "_") + ".png")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(th_plt, mean_hd_rwd_plt, lw=1, color='green', linestyle="-")
    title = "Threshold-mean_hd_reward"
    plt.title(title)
    plt.ion()
    plt.savefig(SAVE_FIG_DIR + title.replace(" ", "_") + ".png")


def plot_one(bot_num, n_bot):
    decay = 0.95
    th = 0
    rewards = [0]
    rank0 = [0]
    rank1 = [0]
    gnum = [0]
    output0 = [0]
    output1 = [0]
    hard_rewards = [0]
    with open(LOG_DIR + "bot_" + str(bot_num) + ".txt") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            if re.match("epsilon_decay:", line):
                words = re.split(" ", line)
                th = float(words[1])
            if re.match("Rewards:", line):
                words = re.split(" ", line)
                rewards.append(rewards[-1] * decay + (1-decay) * (float(words[1]) + float(words[2])))
            if re.match("My_rank:", line):
                words = re.split(" ", line)
                a0 = float(words[1])
                a1 = float(words[2])
                hd = 0
                rank0.append(a0)
                rank1.append(a1)
                if a0 == 0:
                    hd += n_bot
                if a1 == n_bot * 2 - 1:
                    hd -= 2
                hard_rewards.append(hard_rewards[-1] * decay + (1-decay) * hd)
            if re.match("Current:", line):
                words = re.split(" ", line)
                gnum.append(float(words[1]))
            if re.match("Output1:", line):
                words = re.split(" ", line)
                output0.append(float(words[1]))
                output1.append(float(words[3]))
    steps = np.linspace(0, len(rewards), len(rewards), False, dtype=np.float)
    rewards = np.array(rewards)
    rank0 = np.array(rank0)
    rank1 = np.array(rank1)
    gnum = np.array(gnum)
    output0 = np.array(output0)
    output1 = np.array(output1)
    hard_rewards = np.array(hard_rewards)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, rewards, lw=1, color='green', linestyle="-")
    title = "Rewards bot_" + str(bot_num)
    plt.title(title)
    plt.ion()
    if not os.path.exists(SAVE_FIG_DIR):
        os.makedirs(SAVE_FIG_DIR)
    plt.savefig(SAVE_FIG_DIR + title.replace(" ", "_") + ".png")
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, hard_rewards, lw=1, color='green', linestyle="-")
    title = "Hard rewards bot_" + str(bot_num)
    plt.title(title)
    plt.ion()
    if not os.path.exists(SAVE_FIG_DIR):
        os.makedirs(SAVE_FIG_DIR)
    plt.savefig(SAVE_FIG_DIR + title.replace(" ", "_") + ".png")
    return th, np.mean(rewards[int(np.size(rewards, axis=0)/2):]), np.mean(hard_rewards)


if __name__ == '__main__':
    plot_all(12, mode="val")