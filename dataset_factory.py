# -*-coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from scipy import misc

import os
from glob import glob
import pdb

slim = tf.contrib.slim

# setups for debug
raw_data_dir = "ILSVRC2015"
data_dir = "VID_15"
tf_record_dir = "tf_records"
data_split_name = "train"
splits_to_sizes = {
    "train":0,
    "val":0,
    "test":0}

# -------------------------
# Dataloader utils
# -------------------------
def get_tf_filenames(tfrecord_dir,split_name,shuffle=False):
    assert split_name in ["train","val","test"]
    tf_filenames = glob(os.path.join(tf_record_dir,"*"))
    tf_filenames = [f for f in tf_filenames if split_name in f]
    if shuffle: # shuffle filenames
        np.random.shuffle(tf_filenames)
    return tf_filenames

def get_dataset(tf_filename):
    """Get dataset tensor given tf_filename.
    Args:
        tf_filename: Path of tfrecord file to read.
    Returns:
        dataset: A `slim.Dataset`.
    """
    reader = tf.TFRecordReader
    # Features in ILSVRC 2015 tfrecords
    keys_to_features = {
        "image/encoded":tf.FixedLenFeature((),tf.string),
        "image/format":tf.FixedLenFeature((),tf.string,default_value="jpeg"),
        "image/object/bbox/xmin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax":tf.VarLenFeature(dtype=tf.float32),
        "image/image_name":tf.FixedLenFeature((),dtype=tf.string)
        }
    items_to_handlers = {
        "image":slim.tfexample_decoder.Image(
            image_key="image/encoded",format_key="image/format",channels=3),
        "object/bbox":slim.tfexample_decoder.BoundingBox(
            ["xmin","xmax","ymin","ymax"],"image/object/bbox/"),
        "image_name":slim.tfexample_decoder.Tensor("image/image_name"),
        }
    items_to_descriptions = {
        "image":"A rgb image",
        "object/bbox":"A list of bounding box",
        "image_name":"File path of this image",
        }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features,items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources=tf_filename,
        reader=reader,
        decoder=decoder,
        num_samples=0, # has nothing to do with reading actually
        items_to_descriptions=items_to_descriptions)
    
    return dataset

def data_provider(dataset):
    """Given a slim.Dataset returns provided data tensor.
    Args:
        dataset: slim.Dataset.
    Returns:
        image,bbox,image_name: tf.tensor object for further preprocessing.
    """
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        shuffle=False,
        num_epochs=1,
        common_queue_capacity=256,
        common_queue_min= 128)
    [image,bbox,image_name] = provider.get(["image","object/bbox","image_name"])
    return image,bbox,image_name

# -------------------------
# Debug utils
# -------------------------
def debug_tfrecord():
    assert data_split_name in ["train","val","test"]
    tf_filenames = glob(os.path.join(tf_record_dir,"*"))
    if data_split_name == "train":
        tf_filenames = [f for f in tf_filenames if "train" in f]
    else:
        tf_filenames = [f for f in tf_filenames if "val" in f]
              
    # debug on tfrecord
    images,_ = decode_from_tfrecord(tf_filenames[0])
    show_anim(images)
    return

def debug_slim_dataset():
    tf_filenames = get_tf_filenames(tf_record_dir,data_split_name)
    dataset = get_dataset(tf_filenames[0])
    image,bbox,image_name = data_provider(dataset)

    images,bboxes = [],[]
    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
            tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        i = 0
        try:
            while not coord.should_stop():
                img,box,name = sess.run([image,bbox,image_name])
                images.append(img)
                bboxes.append(box)
                i+= 1
                print("process ",i)

        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
        coord.join(threads)
    show_anim(images)
    return

def slim_get_dataset(dataset_dir,data_split_name,
    file_pattern="ILSVRC2015_%s_*.tfrecord",
    reader=None):
    """Given a dataset dir and split name returns a Dataset, an example.
    Args:
        dataset_dir: String, the directory where the dataset files are stored.
        data_split_name: A train/test split name.
        file_pattern: The file pattern used for matching the dataset source files.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined is used.
    Returns:
        A `slim.Dataset` class .
    Raises:
        ValueError: If the `data_split_name` is unknown.
    """
    if data_split_name not in ["train","test","val"]:
        raise ValueError("Data split name unknown %s ." % data_split_name)
    
    file_pattern = os.path.join(dataset_dir,file_pattern % data_split_name)

    if reader is None:
        reader = tf.TFRecordReader

    # Features in ILSVRC 2015 tfrecords
    keys_to_features = {
        "image/encoded":tf.FixedLenFeature((),tf.string),
        "image/format":tf.FixedLenFeature((),tf.string,default_value="jpeg"),
        "image/object/bbox/xmin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax":tf.VarLenFeature(dtype=tf.float32),
        "image/image_name":tf.FixedLenFeature((),dtype=tf.string)
        }
    items_to_handlers = {
        "image":slim.tfexample_decoder.Image(
            image_key="image/encoded",format_key="image/format",channels=3),
        "object/bbox":slim.tfexample_decoder.BoundingBox(
            ["xmin","xmax","ymin","ymax"],"image/object/bbox/"),
        "image_name":slim.tfexample_decoder.Tensor("image/image_name"),
        }
    items_to_descriptions = {
        "image":"A rgb image",
        "object/bbox":"A list of bounding box",
        "image_name":"File path of this image",
        }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features,items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=splits_to_sizes[data_split_name],
        items_to_descriptions=items_to_descriptions)
    
    return dataset
    
def decode_from_tfrecord(tf_filename):
    """Load tfrecord files via their file path.
    Args:
        tf_filename: path of tfrecord file to load.
        preprocessing_fn: function to do preprocessing on image and bbox.
    Returns:
        images,bboxes: list of preprocessed images and bboxes.
    """
    if isinstance(tf_filename,str):
        # num_epoch = 1 means read all tf_filenames only once.
        filename_queue = tf.train.string_input_producer([tf_filename],
            num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer(tf_filename,
            num_epochs=1)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    keys_to_features = {
        "image/encoded":tf.FixedLenFeature((),tf.string),
        "image/format":tf.FixedLenFeature((),tf.string,default_value="jpeg"),
        "image/object/bbox/xmin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin":tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax":tf.VarLenFeature(dtype=tf.float32),
        "image/image_name":tf.FixedLenFeature((),dtype=tf.string)
        }
 
    features = tf.parse_single_example(serialized_example,
        features = keys_to_features)
    
    image = tf.image.decode_jpeg(features["image/encoded"],channels=3)
    xmin = tf.cast(features["image/object/bbox/xmin"],tf.float32)
    xmax = tf.cast(features["image/object/bbox/xmax"],tf.float32)
    ymin = tf.cast(features["image/object/bbox/ymin"],tf.float32)
    ymax = tf.cast(features["image/object/bbox/ymax"],tf.float32)

    images = []
    bboxes = []
    with tf.device("/cpu:0"): # use cpu to read data
        with tf.Session() as sess:
            init_op = [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            epoch = 0
            try:
                while not coord.should_stop():
                    # print("images:",epoch)
                    img,x1,x2,y1,y2 = sess.run([image,xmin,xmax,ymin,ymax])
                    bboxes.append((x1,x2,y1,y2))
                    images.append(img)
                    epoch += 1
            except tf.errors.OutOfRangeError:
                # print("Queue running done, epoch limit reached.")
                pass
            finally:
                # ask threads to stop.
                coord.request_stop()

    # wait for threads to finish
    coord.join(threads)
    return images,bboxes


def show_anim(images):
    """Show animation from a list of frames.
    """
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.ion()
    plt.show()

    if isinstance(images[0],str):
        for i,img in enumerate(images):
            img = misc.imread(img)
            plt.imshow(img)
            plt.pause(0.01)
            print("Show anim {}/{}".format(i+1,len(images)))
    elif isinstance(images[0],np.ndarray):
        for i,img in enumerate(images):
            plt.imshow(img)
            plt.pause(0.01)
            print("Show anim {}/{}".format(i+1,len(images)))
            
        
    plt.close()

if __name__ == "__main__":
    debug_slim_dataset()


