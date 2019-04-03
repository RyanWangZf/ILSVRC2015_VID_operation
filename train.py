# -*- coding: utf-8 -*-
import tensorflow as tf
from tqdm import tqdm,trange

import os
import pdb
import time

import dataset_factory

tf.app.flags.DEFINE_string(
    "data_dir","./tf_records",
    "Input directory of tfrecords data.")

tf.app.flags.DEFINE_string(
    "split_name","train",
    "Split name of data, assert in `train`,`val` or `test`.")

tf.app.flags.DEFINE_integer(
    "num_epoch",50,
    "The number of training epochs.")

FLAGS = tf.app.flags.FLAGS

def main(_):
    tf_filenames = dataset_factory.get_tf_filenames(FLAGS.data_dir,
        FLAGS.split_name,shuffle=True)
    
    for i in range(FLAGS.num_epoch):
        print("-" * 30)
        print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        print("Training epoch {}/{}".format(i+1,FLAGS.num_epoch))
        for tf_filename in tqdm(tf_filenames):
            # read tensor from dataset
            dataset = dataset_factory.get_dataset(tf_filename)
            # bbox: [ymin,xmin,ymax,xmax]
            image,bbox,_ = dataset_factory.data_provider(dataset)
            
            # TODO: receive tensor, return tensor.
            # image,bbox = preprocessing(image,bbox)

            # TODO: run tensor to get a training pair (exemplar,instance)
            # exemplar_img,instance_img,regression_target,conf_target = \
            #       dataset_factory.get_pair(image,bbox)

            pdb.set_trace()
    
    return

if __name__ == "__main__":
    tf.app.run()



