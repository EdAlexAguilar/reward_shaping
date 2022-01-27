import argparse as parser

import numpy as np

from reward_shaping.training.train import train


def main(args):
    train_params = {'steps': args.steps,
                    'video_every': int(1e5),    # note: rendering causes trouble with containers, eventually disable it
                    'n_recorded_episodes': 3,
                    'eval_every': min(10000, int(args.steps / 10)),
                    'n_eval_episodes': 10,
                    'checkpoint_every': int(args.steps / 10)}
    train(args.env, args.task, args.reward, train_params, algo=args.algo,
          seed=args.seed if args.seed is not None else np.random.randint(low=0, high=1000000),
          expdir=args.expdir)


if __name__ == "__main__":
    envs = ['cart_pole_obst', 'bipedal_walker', 'lunar_lander']
    parser = parser.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=envs)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--algo", type=str, default="sac")
    parser.add_argument("--expdir", type=str, default=None, help="name of intermediate dir to group experiments")
    args = parser.parse_args()
    main(args)
