def get_files(logdir, regex, fileregex):
    return logdir.glob(f"{regex}/{fileregex}")


def parse_env_task(filepath: str):
    env, task = None, None
    for env_name in ["cart_pole_obst", "bipedal_walker", "lunar_lander", "racecar2", "racecar"]:
        if env_name in filepath:
            env = env_name
            break
    for task_name in ["fixed_height", "forward", "hardcore", "land", "drive_delta", "drive", "follow_delta"]:
        if task_name in filepath:
            task = task_name
            break
    if not env or not task:
        raise ValueError(f"not able to parse env/task in {filepath}")
    return env, task


def parse_reward(filepath: str):
    for reward in ["default", "tltl", "bhnr", "morl_uni", "morl_dec", "hprs", "hrs_pot"]:
        if reward in filepath:
            return reward
    raise ValueError(f"reward not found in {filepath}")