import time

import numpy as np

all_scores = {
    "Safe Driving": {
        "Shaped": {
            "S": 0.74,
            "S+T": 0.74,
            "S+T+C": 0.20,
        },
        "TLTL": {
            "S": 0.38,
            "S+T": 0.32,
            "S+T+C": 0.10,
        },
        "BHNR": {
            "S": 0.00,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "MORL(unif.)": {
            "S": 0.99,
            "S+T": 0.69,
            "S+T+C": 0.32,
        },
        "MORL(decr.)": {
            "S": 0.96,
            "S+T": 0.75,
            "S+T+C": 0.35,
        },
        "HPRS(ours)": {
            "S": 0.97,
            "S+T": 0.73,
            "S+T+C": 0.33,
        },
    },
    "Follow Leading Vehicle": {
        "Shaped": {
            "S": 0.97,
            "S+T": 0.97,
            "S+T+C": 0.34,
        },
        "TLTL": {
            "S": 0.94,
            "S+T": 0.12,
            "S+T+C": 0.06,
        },
        "BHNR": {
            "S": 0.31,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "MORL(unif.)": {
            "S": 0.74,
            "S+T": 0.73,
            "S+T+C": 0.35,
        },
        "MORL(decr.)": {
            "S": 0.82,
            "S+T": 0.81,
            "S+T+C": 0.37,
        },
        "HPRS(ours)": {
            "S": 1.00,
            "S+T": 0.99,
            "S+T+C": 0.46,
        },
    },
    "Lunar Lander": {
        "Shaped": {
            "S": 0.98,
            "S+T": 0.72,
            "S+T+C": 0.72,
        },
        "TLTL": {
            "S": 0.92,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "BHNR": {
            "S": 0.51,
            "S+T": 0.49,
            "S+T+C": 0.49,
        },
        "MORL(unif.)": {
            "S": 0.91,
            "S+T": 0.91,
            "S+T+C": 0.90,
        },
        "MORL(decr.)": {
            "S": 0.94,
            "S+T": 0.91,
            "S+T+C": 0.91,
        },
        "HPRS(ours)": {
            "S": 0.91,
            "S+T": 0.91,
            "S+T+C": 0.89,
        }
    },
    "Bipedal Walker": {
        "Shaped": {
            "S": 0.99,
            "S+T": 0.99,
            "S+T+C": 0.51,
        },
        "TLTL": {
            "S": 0.96,
            "S+T": 0.45,
            "S+T+C": 0.27,
        },
        "BHNR": {
            "S": 0.21,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "MORL(unif.)": {
            "S": 0.40,
            "S+T": 0.40,
            "S+T+C": 0.19,
        },
        "MORL(decr.)": {
            "S": 0.43,
            "S+T": 0.43,
            "S+T+C": 0.20,
        },
        "HPRS(ours)": {
            "S": 0.96,
            "S+T": 0.96,
            "S+T+C": 0.48,
        }
    },
    "Bipedal Walker (Hardcore)": {
        "Shaped": {
            "S": 0.84,
            "S+T": 0.29,
            "S+T+C": 0.17,
        },
        "TLTL": {
            "S": 0.98,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "BHNR": {
            "S": 0.55,
            "S+T": 0.00,
            "S+T+C": 0.00,
        },
        "MORL(unif.)": {
            "S": 0.07,
            "S+T": 0.03,
            "S+T+C": 0.02,
        },
        "MORL(decr.)": {
            "S": 0.06,
            "S+T": 0.03,
            "S+T+C": 0.02,
        },
        "HPRS(ours)": {
            "S": 0.85,
            "S+T": 0.85,
            "S+T+C": 0.44,
        }
    },
}

colors = {
    "Shaped": "#1f77b4",
    "TLTL": "#ff7f0e",
    "BHNR": "#2ca02c",
    "MORL(unif.)": "#d62728",
    "MORL(decr.)": "#9467bd",
    "HPRS(ours)": "#8c564b",
}

# plot the above scores into a bar plot
# where each group of bars is a task, each bar is a reward shaping method,
# and each bar is the score denoted by "S", "S+T", or "S+T+C".
# normalize the performance as relative w.r.t. the performance of Shaped.

import matplotlib.pyplot as plt

save = False
score_to_show = "S+T+C"
tasks = list(all_scores.keys())
shapings = list(all_scores[tasks[0]].keys())

# get the scores
scores = []
for task, methods in all_scores.items():
    bars = []
    for method, results in methods.items():
        bars.append(results[score_to_show])
    scores.append(bars)

# plot the scores
fig, ax = plt.subplots(figsize=(15, 5))
x = list(range(len(scores)))
width = 0.15

for i, task_scores in enumerate(scores):
    for j, score in enumerate(task_scores):
        if shapings[j] == "Shaped":
            continue
        offset = (j - 2.5) * width
        rel_score = -1 + score / scores[i][0]
        if i == 0:
            ax.bar(x[i] + offset, rel_score, width, label=shapings[j], color=colors[shapings[j]])
        else:
            ax.bar(x[i] + offset, rel_score, width, color=colors[shapings[j]])

ax.hlines(0, -0.30, len(tasks), linestyles="dashed", colors="k", label="Shaped")
ax.set_ylabel('S+T+C Score - Relative to Shaped')
ax.set_title('Scores by task and reward shaping method')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.legend()

if save:
    plt.savefig(f"barplot_scores_{time.time()}.png")
else:
    plt.show()
