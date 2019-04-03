# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

from tensorflow.python.ops import array_ops

import pdb

# -----------------------------------------------------
# Bounding box computation
# -----------------------------------------------------
def bbox_resize(bbox_ref,bbox,name=None):
    """Resize bbox based on a reference bbox,
    assuming that the latter is [0,0,1,1] after transform.
    Useful for updating a collection of box after cropping an image.
    """
    with tf.name_scope(name,"bbox_resize"):
        # Translate
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bbox = bbox - v

        # scale
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                        bbox_ref[3] - bbox_ref[1],
                        bbox_ref[2] - bbox_ref[0],
                        bbox_ref[3] - bbox_ref[1]])

        bbox = bbox / s 
        return bbox

def bbox_filter_overlap(label,bbox,
                                threshold=0.5, 
                                assign_negative=False,
                                scope=None):
    """Filter out bounding box based on relative overlap with reference box [0,0,1,1].
    Remove completely bbox, or assign negative labels to the one outside.

    Return:
        label,bbox: Filtered (or newly assigned) elements.
    """
    with tf.name_scope(scope,"bbox_filter",[label,bbox]):
        score = bbox_intersection(tf.constant([0,0,1,1],bbox.dtype),bbox)
        mask = score > threshold
        if assign_negative:
            label = tf.where(mask,label,-label)
        else:
            label = tf.boolean_mask(label,mask)
            bbox = tf.boolean_max(bbox,mask)

    return label,bbox

def bbox_intersection(bbox_ref,bbox,name=None):
    """Compute relative intersection between a reference box and a 
    collection of bounding box. Namely, compute the quotient between
    intersection area and box area.

    Args:
        bbox_ref: (N,4) or (4,) Tensor with reference bounding box.
        bbox: (N,4) Tensor, collection of bounding box.
    Return:
        (N,) Tensor with relative intersection
    """
    with tf.name_scope(name,"bbox_intersection"):
        bbox = tf.transpose(bbox)
        bbox_ref = tf.transpose(bbox)

    return

def bbox_coord_range(image,bbox):
    """tf.py_func utils, range bbox coordinates to (0,1).
    """
    def func(image,bbox):
        bbox = bbox / image.shape[0]
        return bbox

    bbox = tf.py_func(func,[image,bbox],tf.float32)
    return bbox

# -----------------------------------------------------
# Image computation
# -----------------------------------------------------
def resize_image(image,size,
                        method=tf.image.ResizeMethod.BILINEAR,
                        align_corners=False):
    """Resize an image.

    Args:
        image: An image Tensor to be resized
        size: `int`, Output shape of the resized image
    Return:
        An image Tensor resized.
    """
    height,width,channels = get_image_dims(image)
    image = tf.expand_dims(image,0)
    image = tf.image.resize_images(image,[size,size],method,align_corners)
    image = tf.reshape(image,tf.stack([size,size,channels]))
    return image

def get_image_dims(image):
    """Get image dimensions given a Tensor image.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]