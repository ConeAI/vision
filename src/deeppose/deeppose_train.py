from deeppose_globals import FLAGS
import deeppose_model
import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
from datetime import datetime
import deeppose_globals
from deeppose_draw import drawPoseOnImage as draw


def main():
    trainSetFileNames = os.path.join(FLAGS.data_dir, FLAGS.trainLabels_fn)
    testSetFileNames = os.path.join(FLAGS.data_dir, FLAGS.testLabels_fn)
    train_dir = os.path.join(FLAGS.data_dir, FLAGS.train_dir)

    if not gfile.Exists(train_dir):
        gfile.MakeDirs(train_dir)

    train(trainSetFileNames, testSetFileNames, train_dir)


def read_my_file_format(filename_queue):
    image_reader = tf.WholeFileReader()
    _, image_data = image_reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(image_data, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->float32.
    labels_ = tf.cast(tf.slice(record_bytes, [0], [deeppose_globals.TotalLabels]), tf.float32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes,
                             [deeppose_globals.TotalLabels], [deeppose_globals.TotalImageBytes]),
                             [FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth])
    # Convert from [depth, height, width] to [height, width, depth].
    #processed_example = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return depth_major, labels_


def input_pipeline(argfilenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(argfilenames, shuffle=True)

    image_file, label = read_my_file_format(filename_queue)

    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size

    example_batch, label_batch = tf.train.shuffle_batch(
        [image_file, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return example_batch, label_batch


def train(trainSetFileNames, testSetFileNames, train_dir):
    with open(trainSetFileNames) as f:
        trainSet = f.read().splitlines()
    with open(testSetFileNames) as f:
        testSet = f.read().splitlines()

    with tf.Graph().as_default():
        # Global step variable for tracking processes.
        global_step = tf.Variable(0, trainable=False)

        # Train and Test Set feeds.
        trainSet_batch, trainLabel_batch = input_pipeline(trainSet, FLAGS.batch_size)
        testSet_batch, testLabel_batch = input_pipeline(testSet, FLAGS.batch_size)

        # Placeholder to switch between train and test sets.
        dataShape = [FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, FLAGS.input_depth]
        labelShape = [FLAGS.batch_size, deeppose_globals.TotalLabels]
        example_batch = tf.placeholder(tf.float32, shape=dataShape)
        label_batch = tf.placeholder(tf.float32, shape=labelShape)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = deeppose_model.inference(example_batch, FLAGS.batch_size)

        # Calculate loss.
        loss, meanPixelError = deeppose_model.loss(logits, label_batch)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters.
        train_op = deeppose_model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        config = tf.ConfigProto(
                        device_count = {'GPU': 0}
                            )

        with tf.Session(config=config) as sess:
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            sess.run(init)

            stepinit = 0
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                stepinit = sess.run(global_step)
            else:
                print("No checkpoint found...")

            summary_writer = tf.train.SummaryWriter(train_dir, graph=sess.graph)

            for step in range(stepinit, FLAGS.max_steps):

                start_time = time.time()
                train_examplebatch, train_labelbatch = sess.run([trainSet_batch, trainLabel_batch])
                feeddict = {example_batch: train_examplebatch,
                            label_batch: train_labelbatch}
                _, PxErrValue = sess.run([train_op, meanPixelError], feed_dict=feeddict)
                duration = time.time() - start_time

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, MeanPixelError = %.1f pixels (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, PxErrValue,
                                         examples_per_sec, sec_per_batch))

                if (step % 50 == 0) and (step != 0):
                    summary_str = sess.run(summary_op, feed_dict=feeddict)
                    summary_writer.add_summary(summary_str, step)

                if (step % 100 == 0) and (step != 0):
                    test_examplebatch, test_labelbatch = sess.run([testSet_batch, testLabel_batch])
                    producedlabels, PxErrValue_Test = sess.run([logits,meanPixelError],
                                             feed_dict={example_batch: test_examplebatch,
                                                        label_batch: test_labelbatch})

                    drawing_dir = os.path.join(FLAGS.data_dir, FLAGS.drawing_dir)
                    draw(test_examplebatch[0,...], producedlabels[0,...], drawing_dir, step/100)
                    print('Test Set MeanPixelError: %.1f pixels' %PxErrValue_Test)


                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()
