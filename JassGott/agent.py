from dataclasses import dataclass
from collections import deque, namedtuple
import torch, torch.nn as nn, torch.nn.functional as F
import random
import os

@dataclass
#container for learning parameters, je nachdem apasse
class CFG:
    num_actions: int = 36
    gamma: float = 0.99
    lr: float = 2e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    min_buffer: int = 5_000
    target_update_every: int = 2000
    max_env_steps: int = 10_000_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000
    device: str = "cuda" if torch.cuda.is_available() else "mps"

Transition = namedtuple("T", "s a r sp done mask mask_p")


class DQN(nn.Module):
    
    def __init__(self, num_actions=36):
        super().__init__()
        self.hist = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), #input (N, 4, 9, 36) outpu (N, 16, 9, 36) with ReLU
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),#input (N, 16, 9, 36) outpu (N, 32, 9, 36) with ReLU
            nn.MaxPool2d((1,2)), #input (N, 16, 9, 36) outpu (N, 32, 9, 18) 
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),#input (N, 32, 9, 18) outpu (N, 32, 9, 18) with ReLU
            nn.MaxPool2d((3, 2)),#input (N, 32, 9, 18) outpu (N, 32, 3, 9)
        )
        self.ptr = nn.Sequential(
            nn.Conv2d(4, 8, (1,3),padding=(0,1)), nn.ReLU(), #input (N, 4, 1, 36) outpu (N, 8, 1, 36) with ReLU
            nn.MaxPool2d((1, 2)), #input (N, 8, 1, 36) outpu (N, 8, 1, 18)
            nn.Conv2d(8,8,(1,3),padding=(0,1)), nn.ReLU(), #input (N, 8, 1, 18) outpu (N, 8, 1, 18) with ReLU
        )
        def mlp(i, o): return nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, o), nn.ReLU())
        self.mlp_hand  = mlp(36,64) #(N,36) -> (N,64)
        self.mlp_mode  = mlp(8,32)
        self.mlp_pts   = mlp(2,16)
        self.mlp_turn  = mlp(4,16)
        self.mlp_round = mlp(9,16)
        self.mlp_geschoben = mlp(2, 16)

        fused_in = (32*3*9) + (8*1*18) + 64 + 32 + 16 + 16 + 16 + 16 #1168
        self.head = nn.Sequential(
            nn.Linear(fused_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )


    def forward(self, hist, ptr, hand, mode, points, turn, rnd, geschoben):
        h1 = self.hist(hist).flatten(1) #input (N, 32, 3, 9) outpu (N, 864)
        h2 = self.ptr(ptr).flatten(1) #input (N, 8, 1, 18) outpu (N, 144)
        v_hand = self.mlp_hand(hand)
        v_mode = self.mlp_mode(mode)
        v_pts = self.mlp_pts(points)
        v_turn = self.mlp_turn(turn)
        v_round = self.mlp_round(rnd)
        v_geschoben = self.mlp_geschoben(geschoben)
        x = torch.cat([h1,h2,v_hand,v_mode, v_pts,v_turn, v_round, v_geschoben], dim=1) #864 + 144 + 64 + 32 + 16 + 16 + 16 = 1152 -> (N, 1152)
        return self.head(x)


# Epsilon langsam chliner mache
class EpsilonScheduler:
    def __init__(self, start, end, decay_steps):
        self.start, self.end, self.decay = start, end, decay_steps
        self.t = 0
    def value(self):
        f = min(1.0, self.t / self.decay)
        return self.start + (self.end - self.start) * f 
    def step(self):
        self.t += 1    


#Epsilon-greedy action selection

def select_action(model, state, valid_mask, eps: float, device: str):

    if random.random() < eps:
        idxs = torch.nonzero(valid_mask, as_tuple=False).squeeze(1).tolist()
        return random.choice(idxs) if idxs else 0
    with torch.no_grad():
        q = model(*[x.unsqueeze(0).to(device).float() for x in state]).squeeze(0)
        q = q.masked_fill(valid_mask.to(device) == 0, float("-inf"))
        return int(torch.argmax(q).item())




#for experience replay trining on last few states gege overfitting, 
# You do this to turn a noisy, correlated stream of gameplay events into a stable, 
# randomized training dataset that lets your DQN learn effectively.
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def __len__(self):
        return len(self.buf)
    def push(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, sp, done, mask, mask_p = zip(*batch)

        def stack_state(states):
            cols = list(zip(*states))
            return tuple(torch.stack(col, dim=0) for col in cols)
        S = stack_state(s)
        SP = stack_state(sp)
        A = torch.tensor(a, dtype=torch.long)
        R = torch.tensor(r, dtype=torch.float32)
        D = torch.tensor(done, dtype=torch.float32)
        M = torch.stack(mask, dim=0).to(torch.bool)
        MP = torch.stack(mask_p, dim=0).to(torch.bool)
        return S, A, R, SP, D, M, MP
    


@torch.no_grad()
def compute_target(policy_net, target_net, sp, mask_p):
    q_online = policy_net(*sp)
    q_online = q_online.masked_fill(mask_p == 0, float("-inf"))
    a_star = torch.argmax(q_online, dim=1, keepdim = True)
    q_target = target_net(*sp)
    return q_target.gather(1, a_star).squeeze(1)  # (N, 1) -> (N,)


def dqn_train(policy_net, target_net, optimizer, loss_fn, batch, gamma: float, device: str):
    (h, p, hand, mode, pts, turn, rnd, ges), a, r, (hp, pp, hand_p, mode_p, pts_p, turn_p, rnd_p, ges_p), d, m, mp = batch

    to = lambda *t: [x.to(device) for x in t]
    
    h, p = h.float(), p.float()  # ensure float type
    hand, mode, pts, turn, rnd, ges = hand.float(), mode.float(), pts.float(), turn.float(), rnd.float(), ges.float()
    hp, pp = hp.float(), pp.float()  # ensure float type
    hand_p, mode_p, pts_p, turn_p, rnd_p, ges_p = hand_p.float(), mode_p.float(), pts_p.float(), turn_p.float(), rnd_p.float(), ges_p.float()

    h, p, hand, mode, pts, turn, rnd, ges = to(h, p, hand, mode, pts, turn, rnd, ges)
    hp, pp, hand_p, mode_p, pts_p, turn_p, rnd_p, ges_p = to(hp, pp, hand_p, mode_p, pts_p, turn_p, rnd_p, ges_p)
    a, r, d, m, mp = to(a, r, d, m, mp)

    q_all = policy_net(h, p, hand, mode, pts, turn, rnd, ges)
    q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)  # (N, 1) -> (N,)

    q_sp = compute_target(policy_net, target_net, (hp, pp, hand_p, mode_p, pts_p, turn_p, rnd_p, ges_p), mp)
    q_target = r + (1 - d) * gamma * q_sp

    optimizer.zero_grad(set_to_none = True)
    loss = loss_fn(q_sa, q_target)
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def soft_update_(target_net, policy_net, tau: float = 0.005):
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.lerp_(param.data, tau)  # target_param = tau * param + (1 - tau) * target_param


def save_checkpoint(path, step, policy_net, target_net, optimizer, replay=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'policy_state_dict': policy_net.state_dict(),
        'target_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_len': len(replay.buf) if replay is not None else None,
    }, path)


def load_checkpoint(path, policy_net, target_net, optimizer):
    d = torch.load(path, map_location='cpu')
    policy_net.load_state_dict(d['policy_state_dict'])
    target_net.load_state_dict(d['target_state_dict'])
    optimizer.load_state_dict(d['optimizer_state_dict'])
    return d.get('step', 0)


def greedy_action(model, state, valid_mask, device: str):
    with torch.no_grad():
        q = model(*[x.unsqueeze(0).to(device).float() for x in state]).squeeze(0)
        q = q.masked_fill(valid_mask.to(device) == 0, float("-inf"))
        return int(torch.argmax(q).item())
    
