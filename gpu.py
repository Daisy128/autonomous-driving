
import tensorflow as tf
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
