from deeppose_globals import FLAGS
import copy
import os
import urllib.request
import sys
import zipfile
import glob
import numpy as np
from scipy.io import loadmat
from os.path import basename as b
from PIL import Image
import deeppose_globals
import deeppose_draw

def main():
    transform_example(2)
    # parse_resize_image_and_labels()


def parse_resize_image_and_labels():
    print('Resizing and packing images and labels to bin files.')
    np.random.seed(1701) # to fix test set

    orimage_dir = os.path.join(FLAGS.data_dir, FLAGS.orimage_dir)
    resized_dir = os.path.join(FLAGS.data_dir, FLAGS.resized_dir)
    jnt_fn = os.path.join(FLAGS.data_dir + FLAGS.joints_file)

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
    for index, img_fn in enumerate(imagelist):
        pose_image = PoseImage.from_filename(img_path, joints[index])
        pose_image.resize((FLAGS.input_size, FLAGS.input_size))

        newFileName = os.path.join(resized_dir, b(img_fn).replace( "jpg", "bin" ))

        im_label_pack = np.concatenate((joints[index, :, :].reshape(deeppose_globals.TotalLabels),
                np.asarray(imgFile).reshape(deeppose_globals.TotalImageBytes)))
        im_label_pack.astype(np.uint8).tofile(newFileName)

        if index in permTest:
            print(newFileName, file=fp_test)
        else:
            print(newFileName, file=fp_train)

        if (index % 100 == 0):
            sys.stdout.write("\r%d done" % index) #"\r" deletes previous line
            sys.stdout.flush()

        #"\r" deletes previous line
        sys.stdout.write("\r")
        sys.stdout.flush()

    print('Done.')


class PoseImage(object):

    @classmethod
    def from_filename(cls, img_path, joints):
        return cls(Image.open(img_path), joints)

    def __init__(self, image, joints):
        self.image = image
        self.im_width, self.im_height = self.image.size
        self.joints = joints

    def show(self):
        deeppose_draw.showPoseOnImage(copy.deepcopy(self.image), copy.deepcopy(self.joints))

    def resize(self, shape):
        self.image = self.image.resize((FLAGS.input_size, FLAGS.input_size), Image.ANTIALIAS)
        self.joints[:, 0] *= shape[0]/float(self.im_width)
        self.joints[:, 1] *= shape[1]/float(self.im_height)
        self.im_width, self.im_height = self.image.size

    def flip(self):
        self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        self.joints[:, 0] = self.im_width - self.joints[:, 0]

    def rotate(self, degrees):
        """Rotates image and joints `degrees` ccw"""
        # Use middle of image as (0, 0)
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

    def save_binary(self, filename):
        im_label_pack = np.concatenate((self.joints[:, :].reshape(deeppose_globals.TotalLabels),
                np.asarray(self.image).reshape(deeppose_globals.TotalImageBytes)))
        im_label_pack.astype(np.uint8).tofile(filename)


def find_bounding_box(joints):
    left = min(joints[:, 0])
    right = max(joints[:, 0])
    bottom = min(joints[:, 1])
    top = max(joints[:, 1])
    print(joints)
    return left, bottom, right, top


def transform_example(index=0):
    orimage_dir = os.path.join(FLAGS.data_dir, FLAGS.orimage_dir)
    jnt_fn = os.path.join(FLAGS.data_dir + FLAGS.joints_file)

    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    invisible_joints = joints[:, :, 2] < 0.5
    joints[invisible_joints] = 0
    joints = joints[...,:2]

    imagelist = sorted(glob.glob(os.path.join(orimage_dir, '*.jpg')))

    pose_image = PoseImage.from_filename(imagelist[index], joints[index])
    pose_image.resize((FLAGS.input_size, FLAGS.input_size))
    transform_image(pose_image)


def transform_image(pose_image):
    """Performs multiple transformations on PoseImage."""
    pose_image.show()
    pose_image.flip()
    pose_image.show()
    pose_image.rotate(30)
    pose_image.show()
    print(find_bounding_box(pose_image.joints))


if __name__ == "__main__":
    main()
