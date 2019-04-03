# -*- coding: utf-8 -*-
"""Image data Augmentation.
"""

import tensorflow as tf
import numpy as np 

import pdb

import utils

# -----------------------------------------------------
# From SiamRPN
# -----------------------------------------------------
def tf_random_stretch():
    # TODO
    return

def tf_random_translate():
    # TODO
    return

# -----------------------------------------------------
# From SSD-Tensorflow
# -----------------------------------------------------
def random_flip_left_right(image,box):
    """Random flip left-right of an image and its bounding box.
    """
    # TODO
    def flip_bbox(bbox):
        """Flip bbox coordinates.
        """
        return

    return


def distort_color(image,bbox=None,color_ordering=0,scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0,1]
        color_ordering: Python int, a type of distortion (valid values: 0-3)
    Returns:
        3-D Tensor color-distorted image on range [0,1]
    Raises:
        ValueError: if color_ordering not in [0,3]
    """
    with tf.name_scope(scope,"distort_color",[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
    return tf.clip_by_value(image,0.0,1.0)

def distort_bounding_box_crop(image,bbox,
            min_object_covered=0.25,
            aspect_ratio_range=(0.6,1.67),
            area_range=(0.1,1.0),
            max_attempts=200,
            clip_bbox=True,
            scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    Args:
        image: A image Tensor [height,width,channels]
        bbox: A bbox Tensor specifying [ymin,xmin,ymax,xmax]
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope,"distorted_bounding_box_crop",[image,bbox]):
        # Each bounding box has shape [1,num of bboxes,4]
        # the coordinates are ordered [ymin,xmin,ymax,xmax]
        bbox_begin,bbox_size,distort_bbox = \
            tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bbox,0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)

        # From (1,1,4) to (4,)
        distort_bbox = distort_bbox[0,0]
        # crop the image to the specified bounding box
        cropped_image = tf.slice(image,bbox_begin,bbox_size)
        # restore the shape since the dynamic slice losses 3rd dimension
        cropped_image.set_shape([None,None,3])

        # update bbox: resize and filter out
        bbox = utils.bbox_resize(distort_bbox,bbox)
        # bbox_filter_overlap()
    return cropped_image,bbox

# -----------------------------------------------------
# Main function
# -----------------------------------------------------
def preprocess_for_train(image,bbox,
                                out_shape,
                                scope="preprocessing_train"):
    """Preprocess the given image for training.
    Args:
        image: A `Tensor` representing an image of 321 x 321 size.
        bbox: A list of `Tensor` representing bounding box in the image, 
            specifying [ymin,xmin,ymax,xmax], range in (0,1)
        out_shape: a `int`, the size of output image after preprocessing, 
    
    Returns:
        A processed image and its object bounding box.
    """
    with tf.name_scope(scope,"",[image,bbox]):
        if image.get_shape().ndims != 3:
            raise ValueError("Input must be of size [height,widht,Channel>0]")
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image,dtype=tf.float32)

        # distort image and bboxes
        image,bbox = distort_bounding_box_crop(image,bbox)

        # resize image to output size
        image = utils.resize_image(image,out_shape,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

    return image,bbox

# -----------------------------------------------------
# Debug Utils
# -----------------------------------------------------
def main():
    import dataset_factory

    tf_filenames = dataset_factory.get_tf_filenames("./tf_records",
        "train",shuffle=True)
    dataset = dataset_factory.get_dataset(tf_filenames[0])
    # bbox: [ymin,xmin,ymax,xmax]
    image,bbox,_ = dataset_factory.data_provider(dataset)

    # bbox coordinates range (0,1) for next processing
    bbox = utils.bbox_coord_range(image,bbox)

    # synthetic processing
    image,bbox = preprocess_for_train(image,bbox,out_shape=312)

    # show the processing result
    img,box = visualize_result(image,bbox)
    show_img_and_bbox(img,box)

def visualize_result(image,bbox):
    """Visualize preprocessing function results given image and bbox Tensor.

    Args:
        image: Tensor
        bbox: Tensor, [ymin,xmin,ymax,xmax]
    Returns:
        Visualized np.ndarry object img and box.
    """
    with tf.Session() as sess:
        init_op = [tf.global_variables_initializer(),
            tf.local_variables_initializer()]
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        i = 0
        try:
            while not coord.should_stop() and i < 1: # see one image
                img,box = sess.run([image,bbox])
                i+= 1

        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
    coord.join(threads)
    return img,box

def show_img_and_bbox(img,box):
    """Given img and box, show them.
    Args:
        box: [ymin,xmin,ymax,xmax], range from (0,1)
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches
    # get box rectangle
    scale = img.shape[0]
    box = box.flatten()

    w,h = int(scale*(box[3]-box[1])), int(scale*(box[2]-box[0]))
    xmin,ymin = int(scale*box[1]), int(scale*box[0])
    # cx,cy = int(scale*(box[1]+box[3])/2), int(scale*(box[2]+box[0])/2)

    plt.imshow(img)
    axis = plt.gca()
    rect = patches.Rectangle((xmin,ymin),w,h,linewidth=1,edgecolor="r",facecolor="none")
    axis.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    main()











