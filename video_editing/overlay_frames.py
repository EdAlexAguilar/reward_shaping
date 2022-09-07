import argparse
import pathlib
import re
import time
from typing import List

import cv2
import imutils as imutils
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def get_diff_mask(frame1, frame2):
    # grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # thresholded difference
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # create mask
    mask = cv2.bitwise_not(thresh)
    return mask


def create_overlay_image_from_files(files: List[pathlib.Path]) -> np.ndarray:
    # create overlay of differences among consecutive frames
    # and combine it with first and last frames to create a kind-of trajectory
    first_frame = None
    last_frame = None

    for i, file in enumerate(files):
        frame = cv2.imread(str(file))
        if i == 0:
            first_frame = frame.copy()  # keep first frame to final overlay
            out = frame.copy()          # img where write intermediate results
            last_frame = frame.copy()   # most recent frame for computing diff with next one
            continue
        if i % args.every != 0:         # consider only the frames "every" a given interval
            continue

        mask = get_diff_mask(frame, last_frame)     # mask: binary map isolating difference area between 2 frames
        last_frame = frame.copy()                   # update most recent frame

        background = np.full(frame.shape, 255, dtype=np.uint8)
        cv2.bitwise_or(frame, background, dst=frame, mask=mask)

        out = cv2.bitwise_and(out, frame, dst=out)

        if args.debug:
            cv2.imshow('out', out)
            cv2.waitKey(0)

    # finally combine with first/last frame
    frames = cv2.addWeighted(src1=first_frame, alpha=0.5, src2=last_frame, beta=0.5, gamma=0.0)
    out = cv2.addWeighted(src1=out, alpha=0.35, src2=frames, beta=0.65, gamma=0.0)
    return out


def main(args):
    for indir in args.indir:

        files = [f for f in indir.glob("frame*png")]
        files = sorted(files, key=lambda file: int(re.findall(r'\d+', file.stem)[0]))

        result_img = create_overlay_image_from_files(files)

        cv2.imshow('result', result_img)
        cv2.waitKey(0)

        if args.save:
            outpath = f"output_{time.time()}.jpeg"
            cv2.imwrite(outpath, result_img)
            print(f"[info] save in {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=pathlib.Path, help="path to dir containing frames", nargs="+", required=True)
    parser.add_argument("--every", type=int, help="use only every k frames", default=1)
    parser.add_argument("-save", action="store_true", help="save output image")
    parser.add_argument("-debug", action="store_true", help="step-by-step visualization")
    args = parser.parse_args()

    main(args)
