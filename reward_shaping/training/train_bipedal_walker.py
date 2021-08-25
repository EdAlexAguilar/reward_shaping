import argparse as parser

from train import train


def main(args):
    env_name = "bipedal_walker"
    train_params = {'steps': args.steps,
                    'video_every': int(args.steps / 10),
                    'n_recorded_episodes': 5,
                    'eval_every': min(10000, int(args.steps / 10)),
                    'n_eval_episodes': 5,
                    'checkpoint_every': int(args.steps / 10)}
    train(env_name, args.task, args.reward, train_params, seed=args.seed)


if __name__ == "__main__":
    tasks = ['forward']
    parser = parser.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=tasks)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
