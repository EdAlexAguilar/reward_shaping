import argparse as parser

from reward_shaping.training.train import train


def main(args):
    train_params = {'steps': args.steps,
                    'video_every': args.video_every,
                    'n_recorded_episodes': args.video_episodes,
                    'eval_every': args.eval_every,
                    'n_eval_episodes': args.eval_episodes,
                    'checkpoint_every': args.checkpoint_every}
    train(args.env, args.task, args.reward, train_params, seed=args.seed)


if __name__ == "__main__":
    envs = ['cart_pole_obst', 'bipedal_walker']
    parser = parser.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, choices=envs)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--eval_every", type=int, default=1e5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--video_every", type=int, default=1e5)
    parser.add_argument("--video_episodes", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=1e5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
