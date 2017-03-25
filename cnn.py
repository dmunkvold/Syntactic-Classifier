from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from training_set_builder import SampleBuilder

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic will be added here

if __name__ == "__main__":
  tf.app.run()
  
  
  def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 32, 64, 1])
    
    # for this project, filters slide over whole rows. width of filter needs to be same as width of input filter.
    # Height is around 2-5 words
    # Input Tensor Shape: [batch_size, 32, 64, 1]
    # Output Tensor Shape: [batch_size, 32, 64, 40]
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=10,
        kernel_size=[5, 64],
        padding="same",
        activation=tf.nn.relu)
    
    conv2 = tf.layers.conv2d(
      inputs = input_layer,
      filters = 10,
      kernel_size = [4, 64],
      padding = "same",
      activation = tf.nn.relu)
    
    conv3 = tf.layers.conv2d(
      inputs = input_layer,
      filters = 10,
      kernel_size = [3, 64],
      padding = "same",
      activation = tf.nn.relu)
    
    
    conv4 = tf.layers.conv2d(
      inputs = input_layer,
      filters = 10,
      kernel_size = [2, 64],
      padding = "same",
      activation = tf.nn.relu)      
  
    # Pooling Layer #1
    #input tensor shape: [batch_size, 32, 64, 40]
    # output tensor shape: []
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[32, 64], strides=0)
    
    #For now, 4/23/17, I will use only one conv and pooling layer
    # Convolutional Layer #2 and Pooling Layer #2
    """
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    """
    # Dense Layer
    pool1_flat = tf.reshape(pool2, [-1, 1 * 1 * 40])
    dense = tf.layers.dense(inputs=pool1_flat, units=40, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
  
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
  
    loss = None
    train_op = None
  
    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
      onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits)
  
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.001,
          optimizer="SGD")
  
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }
  
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)
  
  def main(unused_argv):
    # Load training and eval data
    builder = SampleBuilder()
    train_data, train_labels = builder.load_dataset_from_disk()
    eval_data = train_data[400:]
    eval_labels = train_labels[400:]
    train_data = train_data[:400]
    train_labels = train_labels[:400]
    
    mnist_classifier = learn.Estimator(
          model_fn=cnn_model_fn, model_dir="/tmp/syntactic-classifier_convnet_model")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    
    