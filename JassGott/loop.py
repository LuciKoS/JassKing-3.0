import torch
import torch.nn as nn
import torch.optim as optim
from environment import jassgame   #  mit actuall env replace 

# loop.py (change import)
from agent import CFG, DQN, ReplayBuffer, EpsilonScheduler, select_action, Transition, dqn_train
# then create the nets locally:
policy_net = DQN(CFG.num_actions).to(CFG.device)
target_net = DQN(CFG.num_actions).to(CFG.device)
policy_net.train()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


replay = ReplayBuffer(CFG.buffer_size )
eps = EpsilonScheduler(CFG.eps_start, CFG.eps_end, CFG.eps_decay_steps)
optimizer = optim.Adam(policy_net.parameters(), lr=CFG.lr)
loss_fn = nn.SmoothL1Loss()

state, info = jassgame.reset()

mask = info["mask"]

for step in range(CFG.max_env_steps):
    state_t = tuple(torch.as_tensor(x) for x in state)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    a = select_action(policy_net, state_t, mask_t, eps.value(), CFG.device)
    eps.step()

    sp, r, done, trunc, info = jassgame.zug(a)
    mask_p = info["mask"]

    replay.push(Transition(
        s=tuple(torch.as_tensor(x) for x in state),
        a=a,
        r=float(r),
        sp=tuple(torch.as_tensor(x) for x in sp),
        done=float(done or trunc),
        mask=torch.as_tensor(mask, dtype=torch.bool),
        mask_p=torch.as_tensor(mask_p, dtype=torch.bool)
    ))

    if len(replay) >= CFG.min_buffer and step % 4 == 0:
        batch = replay.sample(CFG.batch_size)
        loss = dqn_train(policy_net, target_net, optimizer, loss_fn, batch, CFG.gamma, CFG.device)

    if step % CFG.target_update_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    state, mask = (sp, mask_p)
    if done or trunc:
        state, info = jassgame.reset()
        mask = info["mask"]
