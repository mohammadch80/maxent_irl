import numpy as np
import local_utils as utils
from tqdm import tqdm


def compute_state_visition_freq(P_a, trajs, policy, T):
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 
  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for s in range(N_STATES):
    for t in range(T-1):
      mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  return p


def compute_local_action_probs(P_a, rewards, terminals, iters):
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  z_si = np.zeros([N_STATES])
  z_aij = np.zeros([N_STATES, N_ACTIONS])

  for t in terminals:
    z_si[t] = 1
  for _ in range(iters):
    for s in range(N_STATES):
      for a in range(N_ACTIONS):
        z_aij[s, a] = sum([P_a[s, k, a]*np.exp(rewards[s])*z_si[k] for k in range(N_STATES)])
    for s in range(N_STATES):
      z_si[s] = np.sum(z_aij[s,:])
  
  policy = np.zeros([N_STATES, N_ACTIONS])

  for s in range(N_STATES):
    for a in range(N_ACTIONS):
      if z_si[s] == 0:
        policy[s, a] = 0
      else:
        policy[s, a] = z_aij[s, a] / z_si[s]
  return policy


def maxent_irl(feat_map, P_a, trajs, lr, n_iters, time_horizon, title):
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init parameters
  theta = np.random.uniform(size=(feat_map.shape[1],))

  # calc feature expectations
  feat_exp = np.zeros([feat_map.shape[1]])
  terminals = set([])
  for episode in trajs:
    terminals.add(episode[-1].cur_state)
    for i in range(len(episode)):
      feat_exp += feat_map[episode[i].cur_state,:]
  feat_exp = feat_exp/len(trajs)
  # print(feat_exp)
    
  max_traj_len = 0
  for traj in trajs:
    if len(traj) > max_traj_len:
      max_traj_len = len(traj)

  errors = []
  # training
  for iteration in tqdm(range(n_iters)):
  
    # compute reward function
    rewards = np.dot(feat_map, theta)

    # compute policy
    policy = compute_local_action_probs(P_a, rewards, terminals=terminals, iters=time_horizon)
    
    # compute state visition frequences
    svf = compute_state_visition_freq(P_a, trajs, policy, T=time_horizon)
    
    # compute gradients
    grad = feat_exp - feat_map.T.dot(svf)
    errors.append(np.linalg.norm(grad))
    # update params
    theta += lr * grad

  utils.plot_figure(np.array(errors), "2-Norm of gradient", f"{title}-grads")
  rewards = np.dot(feat_map, theta)
  return utils.normalize(rewards)


