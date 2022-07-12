import re

def get_files(logdir, regex, fileregex):
    return logdir.glob(f"{regex}/{fileregex}")


def parse_env_task(filepath: str):
    env, task = None, None
    for env_name in ["cart_pole_obst", "bipedal_walker", "lunar_lander", "racecar"]:
        if env_name in filepath:
            env = env_name
            break
    for task_name in ["fixed_height", "forward", "hardcore", "land", "drive"]:
        if task_name in filepath:
            task = task_name
            break
    if not env or not task:
        raise ValueError(f"not able to parse env/task in {filepath}")
    return env, task


def parse_reward(filepath: str):
    # note: important to have morl after morl_uni, morl_dec to avoid wrong parsing
    for reward in ["default", "tltl", "bhnr", "morl_uni", "morl_dec", "hprs", "morl_lambda-[0-9]+\.[0-9]*"]:
        res = re.search(reward, filepath)
        if res is not None:
            return res.group(0)
    raise ValueError(f"reward not found in {filepath}")