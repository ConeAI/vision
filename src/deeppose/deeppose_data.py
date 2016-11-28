from deeppose_globals import FLAGS
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

def main():
    parse_resize_image_and_labels()


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
        imgFile = Image.open(img_fn)
        (imWidth, imHeight) = imgFile.size
        imgFile = imgFile.resize((FLAGS.input_size, FLAGS.input_size), Image.ANTIALIAS)
        newFileName = os.path.join(resized_dir, b(img_fn).replace( "jpg", "bin" ))

        joints[index, :, 0] *= FLAGS.input_size/float(imWidth)
        joints[index, :, 1] *= FLAGS.input_size/float(imHeight)

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


if __name__ == "__main__":
    main()
