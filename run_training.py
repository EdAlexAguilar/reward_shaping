import argparse as parser

import numpy as np

from reward_shaping.training.train import train


def main(args):
    video_every = (args.steps - 1) if args.env == "f1tenth" else int(1e5)  # f1tenth records only once at the end
    train_params = {'steps': args.steps,
                    'video_every': video_every,  # note: rendering causes trouble with containers, eventually disable it
                    'n_recorded_episodes': 3,
                    'eval_every': min(10000, int(args.steps / 10)),
                    'n_eval_episodes': 10,
                    'checkpoint_every': int(args.steps / 10)}
    for seed in range(args.n_seeds):
        train(args.env, args.task, args.reward, train_params, algo=args.algo,
              seed=np.random.randint(low=0, high=1000000),
              expdir=args.expdir)


if __name__ == "__main__":
    envs = ['cart_pole_obst', 'bipedal_walker', 'lunar_lander', 'f1tenth']
    parser = parser.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=envs)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--expdir", type=str, default=None, help="name of intermediate dir to group experiments")
    args = parser.parse_args()
    main(args)
