import tensorflow as tf
import numpy as np

tfrecord_file = 'example.tfrecord'

keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'height': tf.FixedLenFeature([], tf.int64),
                    'channels': tf.FixedLenFeature([], tf.int64)}


def _parse_fn(data_record):
    features = keys_to_features
    sample = tf.parse_single_example(data_record, features)
    #channels = sample['channels']
    images = tf.cast(tf.image.decode_jpeg(sample['image']), dtype=tf.float32)
    #images = tf.image.decode_jpeg(sample['image'])
    #images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    #width = sample['width']
    #height = sample['height']
    #channels = sample['channels']
    #images.set_shape([width, height, channels])
    labels = sample['label']
    return images, labels
    #return sample

dataset = tf.data.TFRecordDataset(['example.tfrecord'])
dataset = dataset.map(_parse_fn)
dataset = dataset.batch(20)
dataset = dataset.shuffle(2000)
dataset = dataset.repeat(10)

iterator = dataset.make_one_shot_iterator()
#next_elements = iterator.get_next()
images, labels = iterator.get_next()
print(images, labels)
#print(next_elements)

i = 0
with tf.Session() as sess:
    try:
        while True:
            i += 1
            image, label = sess.run([images, labels])
            #print(type(data['image']))
            print(i)
            #print(image.shape, label)
            '''data = sess.run(next_elements)
            print(data['image'][0])'''
    except:
        print('Training process finished')
        pass
