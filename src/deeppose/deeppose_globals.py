import tensorflow as tf
import os.path as pt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('joints_file', 'joints.mat', """Name of extracted joints file""")

tf.app.flags.DEFINE_string('data_dir', pt.expanduser('~/Desktop/LSP_data/'),
        """Folder for downloading and extracting data.""")
tf.app.flags.DEFINE_string('orimage_dir', pt.expanduser('images'),
        """Folder for saving resized train files""")
tf.app.flags.DEFINE_string('resized_dir', pt.expanduser('resized_images'),
        """Folder for saving resized train files""")
tf.app.flags.DEFINE_string('train_dir', pt.expanduser('train'),
        """Folder for saving train files""")
tf.app.flags.DEFINE_string('drawing_dir', pt.expanduser('train/drawings'),
        """Folder for saving train eval output""")
tf.app.flags.DEFINE_string('input_dir', pt.expanduser('eval'),
        """Input images directory.""")
tf.app.flags.DEFINE_string('output_dir', pt.expanduser('eval/drawings'),
        """Output images directory.""")

tf.app.flags.DEFINE_string('trainLabels_fn', 'train_joints.csv', """Train labels file.""")
tf.app.flags.DEFINE_string('testLabels_fn', 'test_joints.csv', """Test labels file.""")

tf.app.flags.DEFINE_integer('input_size', 224, """One side of CNN's input image size""")
tf.app.flags.DEFINE_integer('input_depth', 3, """Color component size CNN's input image""")
tf.app.flags.DEFINE_integer('label_count', 14, """Label count of images""")
tf.app.flags.DEFINE_integer('label_size', 2, """Label size of images""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Batch size for train""")
tf.app.flags.DEFINE_integer('eval_size', 1, """Batch size for eval""")

tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_float('learn_rate', 0.0005, """Initial learning rate""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Decay to be used for moving average""")

TotalImageBytes = FLAGS.input_size * FLAGS.input_size * FLAGS.input_depth
TotalLabels = FLAGS.label_count * FLAGS.label_size

TOWER_NAME = '-----------'

class BodyParts:
    Right_ankle = 0
    Right_knee = 1
    Right_hip = 2
    Left_hip = 3
    Left_knee = 4
    Left_ankle = 5
    Right_wrist = 6
    Right_elbow = 7
    Right_shoulder = 8
    Left_shoulder = 9
    Left_elbow = 10
    Left_wrist = 11
    Neck = 12
    Head_top = 13

