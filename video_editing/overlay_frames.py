import argparse
import pathlib
import re
import time

import cv2
import imutils as imutils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def main(args):
    indir = args.indir

    files = [str(f) for f in indir.glob("frame*png")]
    files = sorted(files, key=lambda file: re.findall(r'\d+', file))

    first_frame = None
    last_frame = None

    for i, framepath in enumerate(files):
        frame = cv2.imread(str(framepath))
        if i == 0:
            first_frame = frame.copy()
            out = frame.copy()
            last_frame = frame.copy()
            continue
        if i % args.every != 0:
            continue

        grayA = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        last_frame = frame.copy()

        mask = cv2.bitwise_not(thresh)

        background = np.full(frame.shape, 255, dtype=np.uint8)
        cv2.bitwise_or(frame, background, dst=frame, mask=mask)

        out = cv2.bitwise_and(out, frame, dst=out)

        #cv2.imshow('out', out)
        #cv2.waitKey(0)

    frames = cv2.addWeighted(src1=first_frame, alpha=0.5, src2=last_frame, beta=0.5, gamma=0.0)
    out = cv2.addWeighted(src1=out, alpha=0.35, src2=frames, beta=0.65, gamma=0.0)
    #out = cv2.bitwise_and(out, frame, dst=out)

    cv2.imshow('out', out)
    cv2.waitKey(0)

    if args.save:
        outpath = f"output_{time.time()}.jpeg"
        cv2.imwrite(outpath, out)
        print(f"[info] save in {outpath}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=pathlib.Path, help="path to dir containing frames", required=True)
    parser.add_argument("--every", type=int, help="use only every k frames", default=1)
    parser.add_argument("-save", action="store_true", help="save output image")
    args = parser.parse_args()

    main(args)
