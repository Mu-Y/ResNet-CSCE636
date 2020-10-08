import numpy as np
import pdb

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    # depth_major = tf.reshape(record, [3, 32, 32])
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    # image = tf.transpose(depth_major, [1, 2, 0])
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        # image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)
        image = np.pad(image, pad_width=((4, 4), (4, 4), (0, 0)), mode='constant',\
                        constant_values=((0,0),(0,0), (0,0)))


        ### END CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # image = tf.random_crop(image, [32, 32, 3])
        # HINT: randomly generate the upper left point of the image
        row_idx = np.random.randint(low=0, high=8)
        col_idx = np.random.randint(low=0, high=8)
        image = image[row_idx:row_idx+32, col_idx:col_idx+32, :]


        ### END CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        # image = tf.image.random_flip_left_right(image)
        random_num = np.random.uniform()
        if random_num > 0.5:
            image = np.flip(image, axis=1)


        ### END CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    # image = tf.image.per_image_standardization(image)
    x_mean = np.mean(image)
    x_std = max(np.std(image), 1/(np.sqrt(image.shape[0]+image.shape[1]+image.shape[2])))
    image = (image - x_mean) / x_std


    ### END CODE HERE

    return image

