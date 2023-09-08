import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

Step = namedtuple('Step','cur_state action')

save_address = 'figures/'

def normalize(vals):
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def create_arrows(H, W, policy, dirs):
  arrows = {'l': "←", 'd': "↓", 'r': "→", 'u': "↑", 's': "*"}
  arrow_table = np.empty([H, W], dtype=object)

  for r in range(H):
      for c in range(W):
          arrow_table[r, c] = arrows[dirs[policy[r, c]]]

  return arrow_table

def plot_values_policy_map(values, policy, dirs, fig, ax, title):
  H, W = policy.shape
  policy_arrows = create_arrows(H, W, policy, dirs)

  sns.heatmap(
      values,
      annot=policy_arrows,
      fmt="",
      ax=ax,
      cmap=sns.color_palette("Blues", as_cmap=True),
      linewidths=0.7,
      linecolor="black",
      xticklabels=[],
      yticklabels=[],
      annot_kws={"fontsize": "xx-large"},
  ).set(title=title+'-values_policy')
  for _, spine in ax.spines.items():
      spine.set_visible(True)
      spine.set_linewidth(0.7)
      spine.set_color("black")
  img_title = f"{title}-values_policy.png"
  plt.savefig(save_address + img_title, dpi=300, bbox_inches="tight")


def round_rewards(H, W, rewards):
  reward_rounded = np.empty([H, W], dtype=object)

  for r in range(H):
      for c in range(W):
          reward_rounded[r, c] = '%.1f' % rewards[r, c]

  return reward_rounded

def plot_rewards_map(rewards, fig, ax, title):
  H, W = rewards.shape
  rewards = rewards.astype('float')
  rewards_rounded = round_rewards(H, W, rewards)
  sns.heatmap(
        rewards,
        annot=rewards_rounded,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
  ).set(title=title+'-rewards')
  for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(0.7)
    spine.set_color("black")
  img_title = f"{title}-rewards.png"
  plt.savefig(save_address + img_title, dpi=300, bbox_inches="tight")

def plot_figure(items, ylable, title):
  plt.figure(dpi=150)
  plt.plot(items)
  plt.xlabel('Iteration')
  plt.ylabel(ylable)
  plt.title(title)
  plt.savefig(save_address + title, dpi=300, bbox_inches="tight")
  plt.show()