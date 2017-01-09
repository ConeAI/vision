"""
Code to manipulate images that contain pose information.
"""
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
from PIL import Image, ImageDraw

from pose.common import common_globals
from pose.common import draw
from pose.common.common_globals import FLAGS


class PoseImage(object):
    """Class to hold and transform an image containing pose information."""

    @classmethod
    def from_filename(cls, img_path, joints):
        """Returns instance of PoseImage from file."""
        return cls(Image.open(img_path), joints)

    @classmethod
    def from_binary(cls, bin_path, shape):
        """Returns instance of PoseImage from binary file."""
        data = np.fromfile(bin_path, np.uint8)
        joints = data[:common_globals.TotalLabels].reshape([FLAGS.label_count, 2]).astype(float)
        image = Image.fromarray(data[common_globals.TotalLabels:].reshape(shape))
        return cls(image, joints)

    def save_binary(self, filename):
        """Saves data to binary file."""
        im_label_pack = np.concatenate((self.joints[:, :].reshape(common_globals.TotalLabels),
                np.asarray(self.image).reshape(common_globals.TotalImageBytes)))
        im_label_pack.astype(np.uint8).tofile(filename)

    def __init__(self, image, joints):
        self.image = image
        self.im_width, self.im_height = self.image.size
        self.joints = joints

    def copy(self):
        """Returns deepcopy of self."""
        return copy.deepcopy(self)

    def show(self):
        """Show image in window."""
        draw.showPoseOnImage(copy.deepcopy(self.image), copy.deepcopy(self.joints))

    @contextlib.contextmanager
    def preserve_val(self, val=[0., 0.]):
        """Maintains particular rows in joints array."""
        zeros = [i for i, p in enumerate(self.joints) if np.array_equal(p, val)]
        yield
        for i in zeros:
            self.joints[i] = copy.deepcopy(val)

    def resize(self, shape):
        """Resize image to new shape."""
        self.image = self.image.resize((FLAGS.input_size, FLAGS.input_size), Image.ANTIALIAS)
        with self.preserve_val(self.joints):
            self.joints[:, 0] *= shape[0]/float(self.im_width)
            self.joints[:, 1] *= shape[1]/float(self.im_height)

        self.im_width, self.im_height = self.image.size

    def flip(self):
        """Flips image."""
        self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)

        with self.preserve_val():
            self.joints[:, 0] = self.im_width - self.joints[:, 0]

    def rotate(self, degrees):
        """Rotates image and joints `degrees` ccw."""
        with self.preserve_val():
            # use middle of image as (0, 0)
            self.joints[:, 0] -= self.im_width / 2.
            self.joints[:, 1] -= self.im_height / 2.

            self.image = self.image.rotate(degrees, expand=True)
            self.im_width, self.im_height = self.image.size
            rad = np.deg2rad(degrees)

            # Rotate
            rot_matrix = np.matrix([
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)]
            ])
            self.joints = np.array(np.matrix(self.joints) * rot_matrix)

            # Transpose back
            self.joints[:, 0] += self.im_width / 2.
            self.joints[:, 1] += self.im_height / 2.

    def rotate_and_crop(self, degrees):
        """Rotates and crops image to maintain shape."""
        pre_width = self.im_width
        pre_height = self.im_height

        self.rotate(degrees)

        mid_x = self.im_width / 2.
        mid_y = self.im_height / 2.
        x_end = mid_x + pre_width / 2.
        x_start = x_end - pre_width
        y_end = mid_y + pre_height / 2.
        y_start = y_end - pre_height

        self.crop([x_start, y_start, x_end, y_end])

    def crop(self, rect):
        """Crops image to given rectangle."""
        self.image = self.image.crop(rect)
        self.im_width, self.im_height = self.image.size

        with self.preserve_val():
            self.joints[:, 0] -= rect[0]
            self.joints[:, 1] -= rect[1]

    def random_crop_square(self, side_length, bound_square):
        """Crops image with random translation, making sure to include bound_square."""
        assert side_length <= self.im_height and side_length <= self.im_width, \
                'side_length must be smaller than width and height.'
        assert side_length >= bound_square[2] - bound_square[0] and \
                side_length >= bound_square[3] - bound_square[1], \
                'side_length must be larger than bound_square width and height'

        hor_space = side_length - (bound_square[2] - bound_square[0])
        ver_space = side_length - (bound_square[3] - bound_square[1])
        hor_shift = hor_space * random.random()
        ver_shift = ver_space * random.random()

        x_end = hor_shift + bound_square[2]
        x_start = x_end - side_length
        y_end = ver_shift + bound_square[3]
        y_start = y_end - side_length

        if x_start < 0:
            x_start = 0
            x_end = side_length
        elif x_end > self.im_width:
            x_start = self.im_width - side_length
            x_end = self.im_width

        if y_start < 0:
            y_start = 0
            y_end = side_length
        elif y_end > self.im_height:
            y_start = self.im_height - side_length
            y_end = self.im_height

        self.crop([x_start, y_start, x_end, y_end])


def bounding_rect(joints):
    """Returns smallest rectangle that holds all non_zero joints."""
    non_zero = np.array([joint for joint in joints if not np.array_equal(joint, [0., 0.])])
    left = min(non_zero[:, 0])
    right = max(non_zero[:, 0])
    bottom = min(non_zero[:, 1])
    top = max(non_zero[:, 1])
    return [left, bottom, right, top]


def scale_square(rect, scale):
    """Scales rectangle in image and converts to square."""
    # use middle of image as (0, 0)
    mid_x, mid_y = (rect[2] + rect[0]) / 2., (rect[3] + rect[1]) / 2.
    rect[0] -= mid_x
    rect[2] -= mid_x
    rect[1] -= mid_y
    rect[3] -= mid_y

    rect = [c * scale for c in rect]

    # Change to square
    if rect[3] > rect[2]:
        square = [rect[1], rect[1], rect[3], rect[3]]
    else:
        square = [rect[0], rect[0], rect[2], rect[2]]

    # Transpose back
    square[0] += mid_x
    square[2] += mid_x
    square[1] += mid_y
    square[3] += mid_y

    return square
