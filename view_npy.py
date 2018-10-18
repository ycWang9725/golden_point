import numpy as np

Q_TABLE_DIR = "q_table/"
load_path = "bot_0.npy"
npy = np.load(Q_TABLE_DIR + load_path)
npy_state = np.sum(npy, axis=1).reshape((-1,1))
print(0)