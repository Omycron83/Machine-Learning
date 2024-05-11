import bppo, buffer, critic
import torch
import numpy as np
import os
import datetime
import pytz
import gym
import d4rl

def run_d4rl(env_name, gamma = 0.99, v_steps = 2e6, v_ann_dim=512, v_depth=3, v_lr=1e-4, v_batchsize=512, q_bc_steps=2e6, q_ann_dim=1024, q_depth=2, q_lr=1e-4, q_batchsize=512, q_update_freq=2, q_tau=0.05, bc_steps=5e5, bc_ann_dim=1024, bc_depth=2, bc_lr=1e-4, bc_batchsize=512, bppo_steps=1e3, bppo_ann_dim=1024, bppo_depth=2, bppo_lr=1e-4, bppo_batchsize=512, clip_ratio=0.25, entropy_weight=0.0, decay=0.96, omega=0.9):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device(f"cuda:{torch.cude.device(0)}" if torch.cuda.is_available() else "cpu")
    

    #Initializing the replay buffer with the offline dataset
    dataset = env.get_dataset()
    replay_buffer = buffer.OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']))
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(gamma)
    mean, std = replay_buffer.normalize_state()

    value = critic.ValueLearner(device, state_dim, v_ann_dim, v_depth, v_lr, v_batchsize)
    Q_bc = critic.QSarsaLearner(device, state_dim, action_dim, q_ann_dim, q_depth, q_lr, q_update_freq, q_tau, gamma, q_batchsize)
    bc = bppo.BehaviorCloning(device, state_dim, bc_ann_dim, bc_depth, action_dim, bc_lr, bc_batchsize)
    bppo_algorithm = bppo.BehaviorProximalPolicyOptimization(device, state_dim, bppo_ann_dim, bppo_depth, action_dim, bppo_lr, clip_ratio, entropy_weight, decay, omega, bppo_batchsize)


    #Offline Training for value, q-function, bc and bppo 

    v_path = os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'value.pt')
    if os.path.exists(v_path):
        value.load(v_path)
    else:
        for i in range(v_steps):
            value_loss = value.update(replay_buffer)
            if i % 1000 == 0:
                print(f"Step: {i}, Loss: {value_loss:.4f}")

        value.save(v_path)

    q_path = os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'q_bc.pt')
    if os.path.exists(q_path):
        Q_bc.load(q_path)
    else:
        for i in range(q_bc_steps):
            q_bc_loss = Q_bc.update(replay_buffer)
            if i % 1000 == 0:
                print(f"Step: {i}, Loss: {q_bc_loss:.4f}")

        Q_bc.save(q_path)

    best_bc_path = os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'bc_best.pt')
    if os.path.exists(best_bc_path):
        bc.load(best_bc_path)
    else:
        best_bc_score = 0    
        for step in range(int(bc_steps)):
            bc_loss = bc.update(replay_buffer)

            if step % int(100) == 0:
                current_bc_score = bc.offline_evaluate(env, 1, mean, std)
                if current_bc_score > best_bc_score:
                    best_bc_score = current_bc_score
                    bc.save(best_bc_path)
                    np.savetxt(os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'best_bc.csv'), [best_bc_score], fmt='%f', delimiter=',')
                print(f"Step: {step}, Loss: {bc_loss:.4f}, Score: {current_bc_score:.4f}")
                

        bc.save(os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'bc_last.pt'))
        bc.load(best_bc_path)
    
    bppo_path = os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', 'value.pt')
    bppo_algorithm.load(best_bc_path)

    cet_timezone = pytz.timezone('CET')
    current_time = datetime.datetime.now(tz=cet_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
    best_bppo_path = os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', current_time, 'bppo_best.pt')
    Q = Q_bc

    best_bppo_score = bppo_algorithm.offline_evaluate(env, 0, mean, std)
    print('best_bppo_score:',best_bppo_score,'-------------------------')

    for step in range(int(bppo_steps)):
        if step > 200:
            is_clip_decay = False
            is_bppo_lr_decay = False
        bppo_loss = bppo_algorithm.update(replay_buffer, Q, value, is_clip_decay, is_bppo_lr_decay)
        current_bppo_score = bppo_algorithm.offline_evaluate(env, 0, mean, std)

        if current_bppo_score > best_bppo_score:
            best_bppo_score = current_bppo_score
            print('best_bppo_score:',best_bppo_score,'-------------------------')
            bppo.save(best_bppo_path)
            np.savetxt(os.path.join('D:/Damian/PC/Python/ML/RL/Projects/BLL/BPPOFiles', current_time, 'best_bppo.csv'), [best_bppo_score], fmt='%f', delimiter=',')
            bppo_algorithm.set_old_policy()

        print(f"Step: {step}, Loss: {bppo_loss:.4f}, Score: {current_bppo_score:.4f}")

    #Online finetuning:
    bppo_algorithm.
if __name__ == "__main__":
    run_d4rl(env_name="hopper-medium-v2")

