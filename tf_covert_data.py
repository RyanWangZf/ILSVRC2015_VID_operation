# -*- coding: utf-8 -*-
"""
Convert the processed video images into tf_records files.
One trajectory one file, because one video can contain two or more objects.
"""

import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
import matplotlib.pyplot as plt

import os
from glob import glob
import pickle
import pdb

slim = tf.contrib.slim

data_dir = "VID_15" # path of pre-processed video images data
output_dir = "tf_records" # path of output tf records files

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # get processed video path
    video_names = glob(data_dir + "/*")
    video_names = [x for x in video_names if os.path.isdir(x)]
    
    # read meta data
    meta_data_path = os.path.join(data_dir,"meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path,"rb"))
    meta_data = {x[0]:x[1] for x in meta_data}
    
    # do multiprocessing here
    for i,video_name in enumerate(video_names):
        print(i)
        worker(meta_data,video_name)
        if i == 10:
            break

    # pdb.set_trace()

def worker(meta_data,video_name):
    image_names = glob(video_name + "/*")
    video = video_name.split("/")[-1]
    trajs = meta_data[video]

    for k in trajs.keys():
        # create tf_record_writer for each trajectory
        tf_filename = "{}_traj{}.tfrecord".format(
                            os.path.join(output_dir,video),k)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            start_idx = int(trajs[k][0])
            end_idx = int(trajs[k][-1])
            traj_image_names = sorted(image_names)[start_idx:end_idx+1]
            for image_name in traj_image_names:
                image_data,bbox = process_image(image_name)
                # convert a image into example
                example = convert_to_example(image_data,bbox,
                    image_name.encode("ascii"))
                # add to tf_record
                tfrecord_writer.write(example.SerializeToString())
                # print("[{}]{}".format(video,image_name))
    
    # debug on video
    # show_anim(image_names)    
    # pdb.set_trace()
    # process_image(image_names[0])
        
def process_image(image_name):
    """Process a prcessed image.
    Args:
        image_name: string, path to an processed image.
    Returns:sdsds
        image_buffer: string, JPEG encoding of RGB image.
    """
    # read bytes image data
    image_data = tf.gfile.FastGFile(image_name,"rb").read()

    # parse bbox xmin,xmax,ymin,ymax
    gt_w = float(image_name.split("_")[-2])
    gt_h = float(image_name.split("_")[-1][:-4])
    trkid = image_name.split("_")[-3]
    xmin = int((321-1)/2 - gt_w/2)
    xmax = int((321-1)/2 + gt_w/2)
    ymin = int((321-1)/2 - gt_h/2)
    ymax = int((321-1)/2 + gt_h/2)
    bbox = [xmin,xmax,ymin,ymax]

    # bbox debug
    # img = misc.imread(image_name)
    # box = np.array([160,160,gt_w,gt_h])
    # img = add_box_img(img,box)
    # plt.imshow(img)
    # plt.show()

    return image_data,bbox

def convert_to_example(image_data,bbox,image_name):
    """Build an Example proto for an image example
    Args:
        image_data: string, JPEG encoding of RGB image
        bbox: a bounding box contains a list of four integers:
            specifying [xmin,xmax,ymin,ymax]
        image_name: string, this image path
    Return:
        Example proto
    """
    assert len(bbox) == 4
    xmin,xmax,ymin,ymax = bbox

    image_format = b"JPEG"
    example = tf.train.Example(features=tf.train.Features(feature={
        "image/format":bytes_feature(image_format),
        "image/encoded":bytes_feature(image_data),
        "image/object/bbox/xmin":float_feature(xmin),
        "image/object/bbox/xmax":float_feature(xmax),
        "image/object/bbox/ymin":float_feature(ymin),
        "image/object/bbox/ymax":float_feature(ymax),
        "image/image_name":bytes_feature(image_name)
        }))
    return example


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto
    """
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# -----------------------------------------------------
# Debug Utils
# -----------------------------------------------------

def show_anim(image_names):
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.ion()
    plt.show()
    
    for im in sorted(image_names):
        img = imread(im)
        plt.imshow(img)
        plt.pause(0.04)
    plt.close()

def add_box_img(img,boxes,color=(0,255,0)):
    # boxes (cx,cy,w,h)
    if boxes.ndim == 1:
        boxes = boxes[None,:]

    img = img.copy()
    img_ctx = (img.shape[0] - 1) / 2
    img_cty = (img.shape[1] - 1) / 2
    
    for box in boxes:
        cx,cy,w,h = box
        point_1 = [cx-w/2,cy-h/2]
        point_2 = [cx+w/2,cy+h/2]
        
        point_1[0] = np.clip(point_1[0],0,img.shape[0])
        point_2[0] = np.clip(point_2[0],0,img.shape[0])
        point_1[1] = np.clip(point_1[1],0,img.shape[1])
        point_2[1] = np.clip(point_2[1],0,img.shape[1])

        img = cv2.rectangle(img,(int(point_1[0]),int(point_1[1])),
            (int(point_2[0]),int(point_2[1])),color,2)
    
    return img

if __name__ == "__main__":
    main()






