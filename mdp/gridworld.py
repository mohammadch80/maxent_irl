import numpy as np


class GridWorld(object):


  def __init__(self, grid, terminals, trans_prob=1):
  
    self.height = len(grid)
    self.width = len(grid[0])
    self.n_states = self.height*self.width
    self.terminals = terminals
    self.grid = np.copy(grid)
    self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    self.actions = [0, 1, 2, 3, 4]
    self.n_actions = len(self.actions)
    self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}

    self.trans_prob = trans_prob

    self.valid_states = []
    for i in range(self.height):
      for j in range(self.width):
        if self.grid[i][j] != -1:
          self.valid_states.append((i, j))

  def get_random_position(self):
    valid_states = self.valid_states
    start_idx = np.random.randint(0, len(valid_states))
    start_pos = valid_states[start_idx]
    return start_pos

  def get_reward(self, state):
    if not self.grid[state[0]][state[1]] == -1:
      return float(self.grid[state[0]][state[1]])
    else:
      return 0

  def is_terminal(self, state):
    if state in self.terminals:
      return True
    else:
      return False

  def reset(self, start_pos):
    self._cur_state = start_pos
    self._is_done = self.is_terminal(start_pos)
    return start_pos, None, start_pos, self.get_reward(start_pos), self._is_done

  def step(self, action):
    st_prob = self.get_transition_states_and_probs(self._cur_state, action)
    sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for _, prob in st_prob])
    last_state = self._cur_state
    next_state = st_prob[sampled_idx][0]
    self._cur_state = next_state
    reward = self.get_reward(next_state)
    self._is_done = self.is_terminal(next_state)

    return last_state, action, next_state, reward, self._is_done

  def get_transition_states_and_probs(self, state, action):
    
    if self.grid[state[0]][state[1]] == -1:
      return [(state, 1)]
    
    mov_probs = np.zeros([self.n_actions])
    mov_probs[action] = self.trans_prob
    mov_probs += (1-self.trans_prob)/self.n_actions

    for a in range(self.n_actions - 1):
      if mov_probs[a] == 0:
        continue
      inc = self.neighbors[a]
      nei_s = (state[0] + inc[0], state[1] + inc[1])
      if nei_s[0] < 0 or nei_s[0] >= self.height or \
          nei_s[1] < 0 or nei_s[1] >= self.width or self.grid[nei_s[0]][nei_s[1]] == -1:
        # if the move is invalid, accumulates the prob to the current state
        mov_probs[self.n_actions-1] += mov_probs[a]
        mov_probs[a] = 0

    res = []
    for a in range(self.n_actions):
      if mov_probs[a] != 0:
        inc = self.neighbors[a]
        nei_s = (state[0] + inc[0], state[1] + inc[1])
        res.append((nei_s, mov_probs[a]))
    return res
      
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
    return pos[0] + pos[1] * self.height

  def idx2pos(self, idx):
    return (idx % self.height, int(idx / self.height))
