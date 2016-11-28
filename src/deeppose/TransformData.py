import json
import os
import os.path as pt
from os.path import basename as b

import tensorflow as tf
import numpy as np
from PIL import Image


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', pt.expanduser('~/Desktop/LEEDS_data/'),
        """Folder that holds original images.""")
tf.app.flags.DEFINE_string('annotation_json', pt.expanduser('LEEDS_annotations.json'),
        """JSON annotation file.""")
tf.app.flags.DEFINE_string('trainLabels_fn', 'train_joints.csv', """Train labels file.""")
tf.app.flags.DEFINE_string('testLabels_fn', 'test_joints.csv', """Test labels file.""")
tf.app.flags.DEFINE_string('train_dir', pt.expanduser('~/Desktop/LEEDS_data/TrainData'),
        """Folder for saving train files""")
tf.app.flags.DEFINE_string('input_dir', pt.expanduser('~/Desktop/LEEDS_data/EvalData/'),
        """Input images directory.""")
tf.app.flags.DEFINE_string('output_dir', pt.expanduser('~/Desktop/LEEDS_data/EvalData/Drawings'),
        """Output images directory.""")
tf.app.flags.DEFINE_string('drawing_dir', pt.expanduser('~/Desktop/LEEDS_data/TrainData/Drawings'),
        """Folder for saving train files""")

tf.app.flags.DEFINE_string('input_type', 'jpg',  """Input type.""")

tf.app.flags.DEFINE_integer('input_size', 100, """One side of CNN's input image size""")
tf.app.flags.DEFINE_integer('input_depth', 3, """Color component size CNN's input image""")
tf.app.flags.DEFINE_integer('label_count', 14, """Label count of images""")
tf.app.flags.DEFINE_integer('label_size', 2, """Label size of images""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size for train""")
tf.app.flags.DEFINE_integer('eval_size', 1, """Batch size for eval""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('moving_average_decay', 0.9999, """The decay to use for the moving average""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 50000, """Epochs after which learning rate decays""")
tf.app.flags.DEFINE_integer('learn_decay_factor', 0.9, """Learning rate decay factor""")
tf.app.flags.DEFINE_integer('initial_learn_rate', 0.002, """Initial learning rate""")
tf.app.flags.DEFINE_integer('example_per_epoch', 128, """Number of Examples per Epoch for Train""")


TotalImageBytes = FLAGS.input_size * FLAGS.input_size * FLAGS.input_depth
TotalLabels = FLAGS.label_count * FLAGS.label_size


def main():
    parse_resize_image_and_labels()


def remap_joints(joints):
    new_joints = np.zeros((FLAGS.label_count, FLAGS.label_size), np.float32)
    joint_map = {
    }
    return new_joints


RESIZED_DIR = 'resized_images'


def parse_resize_image_and_labels():
    print('Resizing and packing images and labels to bin files.\n')
    np.random.seed(1701) # to fix test set

    annot_json = os.path.join(FLAGS.data_dir, FLAGS.annotation_json)
    with open(annot_json) as fp:
        annot_list = json.loads(fp.read())['root']

    # Create folders to store resized files
    possible_paths = set(['/'.join(annot['img_paths'].split('/')[:-2]) for annot in annot_list])
    for possible_path in possible_paths:
        resized_dir = os.path.join(FLAGS.data_dir, possible_path, RESIZED_DIR)
        if not os.path.exists(resized_dir):
            os.mkdir(resized_dir)

    for index, annot in enumerate(annot_list):
        if annot['img_paths'] != 'lspet_dataset/images/im00001.jpg':
            continue

        img_file = os.path.join(FLAGS.data_dir, annot['img_paths'])
        base_dir = os.path.join(FLAGS.data_dir, '/'.join(annot['img_paths'].split('/')[:-2]))
        resized_dir = os.path.join(base_dir, RESIZED_DIR)

        img = Image.open(img_file)
        (img_w, img_h) = img.size
        img = img.resize((FLAGS.input_size, FLAGS.input_size), Image.ANTIALIAS)
        new_filename = os.path.join(resized_dir, b(img_file).replace('jpg', 'bin'))

        joints = np.array(annot['joint_self'])
        joints = remap_joints(joints)

        print(joints)
        print(joints[:, 0])
        break


if __name__ == "__main__":
    main()
