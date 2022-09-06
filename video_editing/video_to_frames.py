import argparse
import pathlib
import cv2

def main(args):
    outdir = args.outdir
    outdir.mkdir(exist_ok=True, parents=True)
    vidcap = cv2.VideoCapture(str(args.video))

    success, image = vidcap.read()
    count = 0
    while success:
        frame_outpath = str(outdir / f"frame{count}.png")
        cv2.imwrite(frame_outpath, image)
        success, image = vidcap.read()
        count += 1
    print(f"[info] written {count} frames in {outdir}.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", type=pathlib.Path, default="./output")
    args = parser.parse_args()
    main(args)

