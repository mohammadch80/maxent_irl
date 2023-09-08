import gymnasium as gym
import numpy as np


class FrozenLake(object):

  def __init__(self, grid, terminals, trans_prob=1):

    self.height = len(grid)
    self.width = len(grid[0])
    self.n_states = self.height*self.width
    
    self.map = []
    for r in grid:
      row = ""
      for c in r:
        if c == +1:
          row += "G"
        elif c == -1:
          row += "H"
        else:
          row += "F"
      self.map.append(row)
    self.map[0] = 'S' + self.map[0][1:]
    print(self.map)

    self.terminals = terminals
    self.grid = np.copy(grid)
    self.neighbors = [(0, -1), (+1, 0), (0, +1), (-1, 0)]
    self.actions = [0, 1, 2, 3]
    self.n_actions = len(self.actions)
    self.dirs = {0: 'l', 1: 'd', 2: 'r', 3: 'u'}

    self.is_slippery = (trans_prob != 1)

    self.env = gym.make("FrozenLake-v1", is_slippery=self.is_slippery, render_mode="rgb_array", desc=self.map)


  def get_transition_states_and_probs(self, state, action):
    # if self.is_terminal(state):
    #   return [(state, 1)]
    if self.grid[state[0]][state[1]] == -1:
      return [(state, 1)]
    
    if not self.is_slippery:
      inc = self.neighbors[action]
      nei_s = (state[0] + inc[0], state[1] + inc[1])
      if nei_s[0] >= 0 and nei_s[0] < self.height and nei_s[
              1] >= 0 and nei_s[1] < self.width:
        return [(nei_s, 1)]
      else:
        return [(state, 1)]
    else:
      # [(0, -1), (+1, 0), (0, +1), (-1, 0)]
      mov_probs = np.full((self.n_actions,), 1/3)
      negative_dir = 2 + 2 * (action % 2) - action
      mov_probs[negative_dir] = 0

      not_move_prob = 0
      for a in range(self.n_actions):
        inc = self.neighbors[a]
        nei_s = (state[0] + inc[0], state[1] + inc[1])
        if nei_s[0] < 0 or nei_s[0] >= self.height or \
           nei_s[1] < 0 or nei_s[1] >= self.width:
          # if the move is invalid, accumulates the prob to the current state
          not_move_prob += mov_probs[a]
          mov_probs[a] = 0

      res = []
      for a in range(self.n_actions):
        if mov_probs[a] != 0:
          inc = self.neighbors[a]
          nei_s = (state[0] + inc[0], state[1] + inc[1])
          res.append((nei_s, mov_probs[a]))
      if not_move_prob > 0:
        res.append((state, not_move_prob))
      return res


  def is_terminal(self, state):
    if state in self.terminals:
      return True
    else:
      return False
  
  def get_reward(self, state):
    if not self.grid[state[0]][state[1]] == -1:
      return float(self.grid[state[0]][state[1]])
    else:
      return 0
  
  def get_random_position(self):
    start_idx = np.random.randint(0, self.n_states)
    return self.idx2pos(start_idx)

  def reset(self, start_pos):
    self.env.reset()
    self.env.env.s = self.pos2idx(start_pos)
    self._cur_state = start_pos
    self._is_done = self.is_terminal(start_pos)
    if self.env.env.s != self.pos2idx(start_pos):
      print(start_pos, self.env.env.s)
      raise Exception("Invalid reset!!")
    return start_pos, None, start_pos, self.get_reward(start_pos), self._is_done


  def step(self, action):
    observation, reward, terminated, truncated, self._info = self.env.step(action)
    self._is_done = (terminated or truncated)
    last_state = self._cur_state
    self._cur_state = self.idx2pos(observation)
    return last_state, action, self.idx2pos(observation), reward, self._is_done

  def get_transition_mat(self):
    N_STATES = self.height*self.width
    N_ACTIONS = self.n_actions
    P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
    for si in range(N_STATES):
      posi = self.idx2pos(si)
      for a in range(N_ACTIONS):
        probs = self.get_transition_states_and_probs(posi, a)

        for posj, prob in probs:
          sj = self.pos2idx(posj)
          # Prob of si to sj given action a
          P_a[si, sj, a] = prob
    return P_a


  def pos2idx(self, pos):
    return pos[0] * self.width + pos[1]

  def idx2pos(self, idx):
    return (int(idx / self.width), idx % self.width)
