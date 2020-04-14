import tensorflow as tf
def process_image(images):
    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, (224, 224))
    images /= 255
    return images