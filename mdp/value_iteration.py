import math
import numpy as np


def calculate_values_and_policy(P_a, rewards, gamma, error=0.01, deterministic=True):
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  values = np.zeros([N_STATES])

  # estimate values matrix
  while True:
    values_tmp = values.copy()
    for s in range(N_STATES):
      values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

    if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
      break


  if deterministic:
    policy = np.zeros([N_STATES])
    for s in range(N_STATES):
      if rewards[s] == 0 and values[s] == 0:
        policy[s] = N_ACTIONS - 1
      else:
        policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1]) 
                                  for s1 in range(N_STATES)]) 
                                  for a in range(N_ACTIONS)])

    return values, policy
  else:
    policy = np.zeros([N_STATES, N_ACTIONS])
    for s in range(N_STATES):
      if rewards[s] == 0 and values[s] == 0:
        v_s = np.zeros([N_ACTIONS])
        v_s[-1] = 1
      else:
        v_s = np.array([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
      policy[s,:] = np.transpose(v_s/np.sum(v_s))
    return values, policy




