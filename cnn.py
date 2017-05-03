from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from training_set_builder import SampleBuilder
import sklearn
# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic will be added here

  
  
def cnn_model_fn(features, labels, mode):
    
    # Input Layer
    input_layer = tf.reshape(features, [-1, 16, 64, 1])
    
    # for this project, filters slide over whole rows. width of filter needs to be same as width of input filter.
    # Height is around 2-5 words
    # Input Tensor Shape: [batch_size, 32, 64, 1]
    # Output Tensor Shape: [batch_size, 32, 64, 40]
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[4, 64],
        padding="same",
        activation=tf.nn.relu)
    

    
    conv2 = tf.layers.conv2d(
      inputs = input_layer,
      filters = 20,
      kernel_size = [3, 64],
      padding = "same",
      activation = tf.nn.relu)
    
    
    conv3 = tf.layers.conv2d(
      inputs = input_layer,
      filters = 20,
      kernel_size = [2, 64],
      padding = "same",
      activation = tf.nn.relu)      

    # Pooling Layer #1
    #input tensor shape: [batch_size, 32, 64, 40]

    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[16, 64], strides=2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[16, 64], strides=2)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[16, 64], strides=2)
    
   
    # Dense Layer
    pool1_flat = tf.reshape(pool1, [-1, 1 * 1 * 20])
    pool2_flat = tf.reshape(pool2, [-1, 1 * 1 * 20])
    pool3_flat = tf.reshape(pool3, [-1, 1 * 1 * 20])
    #pool4_flat = tf.reshape(pool4, [-1, 1 * 1 * 2])
    
    #pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[4, 2], strides=2) 
    
    
    dense = tf.layers.dense(inputs=pool1_flat + pool2_flat + pool3_flat, units=1200, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
  
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
            learning_rate=0.1,
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
   
    evaluate()
    
def evaluate():
    
    # Load training and eval data
    builder = SampleBuilder(['difficult','easy'])
    data, labels = builder.load_dataset_from_disk()
    eval_data = np.array(data[400:], dtype = np.float32)
    eval_labels = np.array(labels[400:], dtype = np.int32)
    train_data = np.array(data[:400], dtype = np.float32)
    train_labels = np.array(labels[:400], dtype = np.int32)
    print("Loaded training and eval data.")

    syntactic_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="./model/syntactic-classifier_convnet_model84")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)
            
    print("Set up logging.")
        
    
    # Train the model
    syntactic_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=25,
        steps=400,
        #steps=20000,
        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = syntactic_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results) 
    

if __name__ == "__main__":
    tf.app.run() 
    

#originally, all that is in evaluate was in main, and there was no session or graph
    
    
    