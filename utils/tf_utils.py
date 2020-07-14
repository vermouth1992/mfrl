import tensorflow as tf

ALLOW_GROWTH = False


def set_tf_allow_growth(enable=True):
    global ALLOW_GROWTH
    if enable != ALLOW_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, enable)
        ALLOW_GROWTH = enable
