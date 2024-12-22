from __future__ import absolute_import, division, print_function
import numpy as np
import os
import cv2
import pathlib
from tf_inception_score import get_inception_score


# get image list from the path
def _handle_path(path, resize=False, h=128, w=128):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.bmp')) + list(path.glob('*.JPEG'))

    if resize:
        x = [cv2.resize(cv2.imread(str(fn)), (h, w)).astype(np.float32) for fn in files]
    else:
        x = [cv2.imread(str(fn)).astype(np.float32) for fn in files]
    return x


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, help='Path to the images fold')
    parser.add_argument("--gpu", default="", type=str, help='GPU to use (leave blank for CPU only)')
    parser.add_argument("--resize", action="store_true", help='Resize images to have the same size.')
    parser.add_argument("--height", default=128, type=int, help='Resize height')
    parser.add_argument("--width", default=128, type=int, help='Resize width')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    imgs = _handle_path(args.path, args.resize, args.height, args.width)

    IS = get_inception_score(imgs, splits=1)

    print("IS: ", IS)

