import os, sys, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python_env.env import JassEnv

env = JassEnv()
s, info = env.reset()
print([x.shape for x in s], int(info["mask"].sum()))
a = int(np.flatnonzero(info["mask"])[0])
sp, r, done, trunc, info2 = env.zug(a)
print(r, done, [x.shape for x in sp])