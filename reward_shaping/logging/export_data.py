import argparse
import os
import pathlib

import numpy as np
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# code build on top of: https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv

def process_logdir(path: pathlib.Path):
    # path: this/is/a/logdir/SAC_1/event_bla_bla_bla
    expdir, logdir = path.parts[-4], path.parts[-3]
    return f"{expdir}_{logdir}"


def to_csv(dpath: pathlib.Path, opath: pathlib.Path):
    print(f"Looking for logs in {dpath.absolute}")

    for path in dpath.rglob("**/events*"):
        print(f"Log: {path}", end="")
        ea = EventAccumulator(str(path)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            if not 'eval' in tag:
                # we are only interested on eval metrics
                continue
            tag_values = []
            wall_time = []
            steps = []

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            out[tag] = pd.DataFrame(data=dict(zip(steps, np.array([tag_values, wall_time]).transpose())), columns=steps,
                                    index=['value', 'wall_time'])

        if len(tags) > 0:
            df = pd.concat(out.values(), keys=out.keys())
            outfile = opath / f"{process_logdir(path)}.csv"
            df.to_csv(outfile)
            print("- Done")
        else:
            print('- Not scalers to write')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=pathlib.Path, required=True, help="experiment dir where look for logs")
    args = parser.parse_args()

    inpath = args.path
    outpath = pathlib.Path("exports")
    outpath.mkdir(parents=True, exist_ok=True)

    to_csv(inpath, outpath)
    print(f"\n\nLogs exported in {outpath.absolute()}")
