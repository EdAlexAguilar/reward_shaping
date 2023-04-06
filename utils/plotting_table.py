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
        "HPRS": {
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
        "HPRS": {
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
        "HPRS": {
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
        "HPRS": {
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
        "HPRS": {
            "S": 0.85,
            "S+T": 0.85,
            "S+T+C": 0.44,
        }
    },
}


COLORS = {
    'Shaped': '#377eb8',
    'TLTL': '#4daf4a',
    'BHNR': '#984ea3',
    'MORL(unif.)': '#a65628',
    'MORL(decr.)': '#ff7f00',
    'HPRS': '#e41a1c',
}

HATCHES = {
    'Shaped': '-',
    'TLTL': '\\\\',
    'BHNR': '//',
    'MORL(unif.)': '-',
    'MORL(decr.)': '--',
    'HPRS': 'xx',
}

LARGESIZE, MEDIUMSIZE, SMALLSIZE = 16, 13, 10

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': LARGESIZE})
plt.rcParams.update({'axes.titlesize': LARGESIZE})
plt.rcParams.update({'axes.labelsize': MEDIUMSIZE})
plt.rcParams.update({'xtick.labelsize': MEDIUMSIZE})
plt.rcParams.update({'ytick.labelsize': SMALLSIZE})
plt.rcParams.update({'legend.fontsize': MEDIUMSIZE})
plt.rcParams.update({'figure.titlesize': LARGESIZE})

# plot the above scores into a bar plot
# where each group of bars is a task, each bar is a reward shaping method,
# and each bar is the score denoted by "S", "S+T", or "S+T+C".
# normalize the performance as relative w.r.t. the performance of Shaped.


save = True
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
        offset = (j - len(tasks)/2) * width
        rel_score = -1 + score / scores[i][0]
        if i == 0:
            ax.bar(x[i] + offset, rel_score, width, label=shapings[j], color=COLORS[shapings[j]],
                   edgecolor='black', linewidth=1, hatch=HATCHES[shapings[j]])
        else:
            ax.bar(x[i] + offset, rel_score, width, color=COLORS[shapings[j]],
                   edgecolor='black', linewidth=1, hatch=HATCHES[shapings[j]])

ax.hlines(0, -0.50, len(tasks)-0.40, linestyles="dashed", colors="k", label="Shaped")

ax.set_ylim(-1.7, 1.7)
ax.set_ylabel('Rel. Performance to Shaped (%)')
ax.set_xticks(x)
ax.set_xticklabels(tasks)

# remove box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# place legend outside of the plot, horizontally centered below
# more margin from the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(shapings), bbox_to_anchor=(0.5, 0.0))
plt.subplots_adjust(bottom=0.2)

if save:
    plt.savefig(f"barplot_scores_{time.time()}.pdf", bbox_inches='tight')
else:
    plt.show()
