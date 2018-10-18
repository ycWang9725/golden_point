import numpy as np
import os
import sys


base_dir = os.path.dirname(os.path.abspath(__file__))
ASSETS = "/assets"
MEM_DIR = base_dir + ASSETS + "/IIR_mem/"


class IIR:
    def __init__(self, alpha=0.5, noise_rate=0.01):
        self.alpha = alpha
        self.noise_rate = noise_rate
        if not os.path.exists(MEM_DIR):
            os.makedirs(MEM_DIR)
        if not os.path.exists(MEM_DIR + "mem1.npy"):
            self.mem = 19.5
        else:
            self.mem = np.load(MEM_DIR + "mem1.npy")

    def get_next(self, last_gnum):
        if last_gnum is not None:
            base = self.mem * self.alpha + last_gnum * (1 - self.alpha)
        else:
            base = self.mem
        result = np.array([base + self.noise_rate * np.random.randn(), base + self.noise_rate * np.random.randn()])
        self.mem = base
        np.save(MEM_DIR + "mem1.npy", self.mem)
        return result
