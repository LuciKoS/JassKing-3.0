import torch
import torch.nn as nn
import torch.optim as optim
from python_env.env import jassgame  #  mit actuall env replace 
from JassGott.agent import save_checkpoint
import os

# loop.py (change import)
from JassGott.agent import CFG, DQN, ReplayBuffer, EpsilonScheduler, select_action, Transition, dqn_train
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

start_step = 0

ckpt = "/Users/lucbaumeler/Documents/Eth/VsCode/MLS/RL/JassKing-3.0/checkpoints/policy_29_10.pt"

if os.path.exists(ckpt):
    from JassGott.agent import load_checkpoint
    
    d = torch.load(ckpt, map_location=CFG.device)
    policy_net.load_state_dict(d['policy_state_dict'])
    target_net.load_state_dict(d['target_state_dict'])
    optimizer.load_state_dict(d['optimizer_state_dict'])
    start_step = d.get('step', 0)
    print(f"Resumed from {ckpt} at step {start_step}")
else:
    # fresh start: clone policy→target once
    target_net.load_state_dict(policy_net.state_dict())

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

    if step % 100_000 == 0 and step > 0:
        save_checkpoint(
            path=f"checkpoints/jass_dqn_step_{step}.pt",
            step=step,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            replay=replay,  # optional
        )
    if step % 1000 == 0 :
        print(f"step: {step}")

    if step % CFG.target_update_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    state, mask = (sp, mask_p)
    if done or trunc:
        state, info = jassgame.reset()
        mask = info["mask"]
