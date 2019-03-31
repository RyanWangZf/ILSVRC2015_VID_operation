# -*-coding: utf-8 -*-

import tensorflow as tf
import os
from glob import glob
import pdb

raw_data_dir = "ILSVRC2015"
data_dir = "VID_15"
tf_record_dir = "tf_records"

# assert `train` or `val` or `test`
data_split_name = "train"

def main():
    assert data_split_name in ["train","val","test"]
    tf_filenames = glob(os.path.join(tf_record_dir,"*"))
    if data_split_name == "train":
        tf_filenames = [f for f in tf_filenames if "train" in f]
    else:
        tf_filenames = [f for f in tf_filenames if "val" in f]
        
     
    # debug on tfrecord
    decode_from_tfrecord(tf_filenames[0])

    pdb.set_trace()

# -------------------------
# Debug utils
# -------------------------
def decode_from_tfrecord(tf_filename):
    filename_queue = tf.train.string_input_producer([tf_filename],
        num_epochs=1)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={"image/encoded":tf.FixedLenFeature((),tf.string),})
    
    image = tf.image.decode_jpeg(features["image/encoded"],channels=3)
    # image = tf.decode_raw(features["image/encoded"],tf.uint8)
    # image = tf.reshape(image,[321,321,3])
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                img = sess.run(image)
        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            # ask threads to stop.
            coord.request_stop()

    # wait for threads to finish
    coord.join(threads)
    pdb.set_trace()



if __name__ == "__main__":
    main()


