import sys
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if  not isinstance(value, list):
        value = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _image_to_tfexample(image_data, label, width, height, channels):
    return tf.train.Example(features=tf.train.Features(feature={'image': _bytes_feature(image_data),
                                                                'label': _int64_feature(label),
                                                                'width': _int64_feature(width),
                                                                'height': _int64_feature(height),
                                                                'channels': _int64_feature(channels)}))


image_phl = tf.placeholder(shape=[None, None, None], dtype=tf.uint8)
jpg_encoded = tf.image.encode_jpeg(image_phl, quality=100)

with tf.Session() as sess:
    with tf.python_io.TFRecordWriter('example.tfrecord') as tfrecord_writer:
        for i in range(10000):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, 10000))
            sys.stdout.flush()
            image = np.random.rand(200, 200, 3)
            label = np.random.randint(0, 10)
            width = image.shape[1]
            height = image.shape[0]
            channels = image.shape[2]

            jpg_string = sess.run(jpg_encoded, feed_dict={image_phl: image})
            #print(type(jpg_string))

            example = _image_to_tfexample(jpg_string, label, width, height, channels)

            tfrecord_writer.write(example.SerializeToString())