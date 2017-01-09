import copy
import contextlib
import os
import urllib.request
import random
import sys
import zipfile
import glob

import numpy as np
from scipy.io import loadmat
from os.path import basename as b

from pose.common import common_globals
from pose.common.common_globals import FLAGS
from pose.common.data import PoseImage, bounding_rect, scale_square


def main():
    # transform_example(1)
    parse_resize_image_and_labels()


def parse_resize_image_and_labels():
    print('Resizing and packing images and labels to bin files.')
    np.random.seed(1701) # to fix test set

    orimage_dir = os.path.join(FLAGS.data_dir, FLAGS.orimage_dir)
    resized_dir = os.path.join(FLAGS.data_dir, FLAGS.resized_dir)
    jnt_fn = os.path.join(FLAGS.data_dir + FLAGS.joints_file)

    print(resized_dir)

    if not os.path.exists(resized_dir):
        os.mkdir(resized_dir)

    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[...,:2]

    N_test = int(len(joints) * 0.1)
    permTest = np.random.permutation(int(len(joints)))[:N_test].tolist()

    imagelist = sorted(glob.glob(os.path.join(orimage_dir, '*.jpg')))

    fp_train = open(os.path.join(FLAGS.data_dir, FLAGS.trainLabels_fn), 'w')
    fp_test = open(os.path.join(FLAGS.data_dir, FLAGS.testLabels_fn), 'w')
    for i, img_fn in enumerate(imagelist):
        written_files = transform_and_write(img_fn, joints[i], resized_dir, TRANSFORMS)

        if i in permTest:
            for wf in written_files:
                print(wf, file=fp_test)
        else:
            for wf in written_files:
                print(wf, file=fp_train)

        print('File {} wrote {} transforms'.format(i, len(written_files)))

    print('Done.')


TRANSFORMS = [
    [],
    [{'func': 'rotate', 'args': [90]}],
    [{'func': 'rotate', 'args': [180]}],
    [{'func': 'rotate', 'args': [270]}],
    [{'func': 'flip', 'args': []}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [90]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [180]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [270]}],
    [{'func': 'random_crop_square', 'args': []}],
    [{'func': 'rotate', 'args': [90]}, {'func': 'random_crop_square', 'args': []}],
    [{'func': 'rotate', 'args': [180]}, {'func': 'random_crop_square', 'args': []}],
    [{'func': 'rotate', 'args': [270]}, {'func': 'random_crop_square', 'args': []}],
    [{'func': 'flip', 'args': []}, {'func': 'random_crop_square', 'args': []}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [90]},
        {'func': 'random_crop_square', 'args': []}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [180]},
        {'func': 'random_crop_square', 'args': []}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate', 'args': [270]},
        {'func': 'random_crop_square', 'args': []}],
    [{'func': 'rotate_and_crop', 'args': [30]}],
    [{'func': 'rotate_and_crop', 'args': [60]}],
    [{'func': 'rotate_and_crop', 'args': [-30]}],
    [{'func': 'rotate_and_crop', 'args': [-60]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate_and_crop', 'args': [30]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate_and_crop', 'args': [60]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate_and_crop', 'args': [-30]}],
    [{'func': 'flip', 'args': []}, {'func': 'rotate_and_crop', 'args': [-60]}],
]

DEFAULT_SCALE = 1.5
CROP_RATIO = 0.75
MIN_SIDE_LENGTH_RATIO = 0.67


def transform_and_write(im_path, joints, resized_dir, transforms=[[]], dry_run=False):
    """Performs transforms on image and joints, and writes to binary."""
    written_files = []
    pose_image = PoseImage.from_filename(im_path, joints)
    for transform in transforms:
        cur_image = pose_image.copy()
        should_write = True
        desc = ''
        for obj in transform:
            desc += obj['func'] + ''.join(map(str, obj['args']))

            if obj['func'] == 'random_crop_square':
                bound_sq = scale_square(bounding_rect(cur_image.joints), DEFAULT_SCALE)

                side_length = max(
                    MIN_SIDE_LENGTH_RATIO * FLAGS.input_size,  # Bigger than ratio of input_size
                    (bound_sq[2] - bound_sq[0]) * 1.00001,     # Bigger than bound_sq
                    min(cur_image.im_width, cur_image.im_height) * CROP_RATIO,
                )

                if side_length <= min(cur_image.im_width, cur_image.im_height):
                    cur_image.random_crop_square(side_length, bound_sq)
                else:
                    # No transform would have been done
                    should_write = False

            else:
                getattr(cur_image, obj['func'])(*obj['args'])

            if obj['func'] == 'rotate_and_crop':
                for x, y in [p for p in cur_image.joints if not np.array_equal(p, [0., 0.])]:
                    if x < 0. or x > cur_image.im_width or y < 0. or y > cur_image.im_height:
                        # Joint out of bounds
                        should_write = False

        cur_image.resize((FLAGS.input_size, FLAGS.input_size))

        if should_write:
            new_path = os.path.join(resized_dir, b(im_path).replace( ".jpg", desc + '.bin'))
            written_files.append(new_path)
            if dry_run:
                cur_image.show()
            else:
                cur_image.save_binary(new_path)

    return written_files


def transform_example(index=0):
    orimage_dir = os.path.join(FLAGS.data_dir, FLAGS.orimage_dir)
    jnt_fn = os.path.join(FLAGS.data_dir + FLAGS.joints_file)

    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[...,:2]

    imagelist = sorted(glob.glob(os.path.join(orimage_dir, '*.jpg')))

    print(transform_and_write(imagelist[index], joints[index], 'resized',
                              transforms=TRANSFORMS, dry_run=True))


if __name__ == "__main__":
    main()
