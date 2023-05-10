import torch
import numpy as np

def simulate_policy(policy_function, n=5, seeds=[0]):
    # For each episode save the total days confined and total dead
    tot_days_confined = []
    tot_dead = []
    cum_rewards = []
    for i in range(n):
        tot, _, _, pol, r = policy_function(n_weeks=30, simulation_seed=seeds[i])
        cum_rewards.append(torch.sum(torch.tensor(r)))
        tot_dead.append(tot['dead'][-1])
        tot_days_confined.append(7 * np.sum(pol))
    return tot_days_confined, tot_dead, cum_rewards