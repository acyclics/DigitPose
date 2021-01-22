import numpy as np
from dp_roipool import ROIPoolingLayer
import tensorflow as tf
# Define parameters
batch_size = 1
img_height = 56
img_width = 56
n_channels = 512
n_rois = 1
pooled_height = 7
pooled_width = 7
# Create feature map input
feature_maps_shape = (batch_size, img_height, img_width, n_channels)
feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
feature_maps_np = np.ones(feature_maps_tf.shape, dtype='float32')
feature_maps_np[0, img_height-1, img_width-3, 0] = 50
print(f"feature_maps_np.shape = {feature_maps_np.shape}")
# Create batch size
roiss_tf = tf.placeholder(tf.float32, shape=(None, None, 4))
roiss_np = np.asarray([[[0.5,0.2,0.7,0.4]]], dtype='float32')
print(f"roiss_np.shape = {roiss_np.shape}")
# Create layer
roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
pooled_features = roi_layer([feature_maps_tf, roiss_tf])
has_nan = tf.reduce_mean(tf.cast(tf.math.is_nan(pooled_features), dtype=tf.float32))
print(f"output shape of layer call = {pooled_features.shape}")
# Run tensorflow session
with tf.Session() as session:
    result, check = session.run([pooled_features, has_nan], 
                         feed_dict={feature_maps_tf:feature_maps_np,  
                                    roiss_tf:roiss_np})
    
print("Nan check", check)
print(f"result.shape = {result.shape}")
print(f"first  roi embedding=\n{result[0,0,:,:,0]}")
