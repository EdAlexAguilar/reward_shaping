import argparse as parser

import numpy as np

from reward_shaping.training.train import train


def main(args):
    video_every = (args.steps - 1) if args.env == "f1tenth" else int(args.steps / 10)  # f1tenth only once at the end
    train_params = {'steps': args.steps,
                    'video_every': video_every,  # note: causes trouble with containers, one can disable it wt -novideo
                    'n_recorded_episodes': 3,
                    'eval_every': min(10000, int(args.steps / 10)),
                    'n_eval_episodes': 10,
                    'checkpoint_every': int(args.steps / 10)}
    for seed in range(args.n_seeds):
        train(args.env, args.task, args.reward, train_params, algo=args.algo,
              seed=np.random.randint(low=0, high=1000000),
              expdir=args.expdir,
              novideo=args.novideo)


if __name__ == "__main__":
    envs = ['cart_pole_obst', 'bipedal_walker', 'lunar_lander', 'f1tenth', 'racecar']
    parser = parser.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=envs)
    parser.add_argument("--task", type=str, required=True, help="task executed for the env")
    parser.add_argument("--reward", type=str, required=True, help="identifier of reward definition")
    parser.add_argument("--steps", type=int, default=1e6, help="nr training steps")
    parser.add_argument("--n_seeds", type=int, default=1, help="nr runs, each with a different rnd seed")
    parser.add_argument("--algo", type=str, default="sac", help="rl algorithm used for training")
    parser.add_argument("--expdir", type=str, default=None, help="name of intermediate dir to group experiments")
    parser.add_argument("-novideo", action="store_true", help="disable recording of videos during training")
    args = parser.parse_args()
    main(args)
