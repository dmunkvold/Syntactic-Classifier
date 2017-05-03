import numpy as np
import tensorflow as tf
from training_set_builder import SampleBuilder
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class SyntacticClassifier():
    def __init__(self, savefile):
        self.savefile=savefile
        
        
    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        self.input_layer = tf.reshape(features, [-1, 32, 64, 1])
        
        # for this project, filters slide over whole rows. width of filter needs to be same as width of input filter.
        # Height is around 2-5 words
        # Input Tensor Shape: [batch_size, 32, 64, 1]
        # Output Tensor Shape: [batch_size, 32, 64, 40]
        # Convolutional Layer #1
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=40,
            kernel_size=[5, 64],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        #input tensor shape: [batch_size, 32, 64, 40]
        # output tensor shape: []
        
        #pool1 = tf.layers.max_pooling2d(inputs=tf.add(tf.add(conv1, conv2), tf.add(conv2, conv4)), pool_size=[32, 64], strides=1)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[32, 64], strides=1)
        
        #For now, 4/23/17, I will use only one conv and pooling layer
        # Convolutional Layer #2 and Pooling Layer #2
        # Dense Layer
        self.pool1_flat = tf.reshape(self.pool1, [-1, 1 * 1 * 40])
        dense = tf.layers.dense(inputs=self.pool1_flat, units=40, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=2)
  
        self.loss = None
        self.train_op = None
        
        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            self.onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
            self.loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.onehot_labels, logits=logits)
            
        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            self.train_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")
            
            # Generate Predictions
        self.predictions = {
            "classes": tf.argmax(
                input=logits, axis=1),
            "probabilities": tf.nn.softmax(
                logits, name="softmax_tensor")
        }
                
        # Return a ModelFnOps object
        self.saver = tf.train.Saver()
        return model_fn_lib.ModelFnOps(
            mode=mode, predictions=self.predictions, loss=self.loss, train_op=self.train_op) 
    
    def fit(self):
        
        init = tf.global_variables_initializer()
        
        # Load training and eval data
        builder = SampleBuilder(['difficult','easy'])
        data, labels = builder.load_dataset_from_disk()
        eval_data = np.array(data[400:], dtype=np.float32)
        eval_labels = np.array(labels[400:], dtype=np.int32)
        train_data = np.array(data[:400], dtype = np.float32)
        train_labels = np.array(labels[:400], dtype=np.int32)
        
        print("Loaded training and eval data.")
        
        syntactic_classifier = learn.Estimator(
            model_fn=self.cnn_model_fn, model_dir="./model/syntactic-classifier_convnet_model")
        init = tf.global_variables_initializer()
        # Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        
        print("Set up logging.")
                
        with tf.Session() as session:
            
            # Train the model
            syntactic_classifier.fit(
                x=train_data,
                y=train_labels,
                batch_size=25,
                steps=1,
                #steps=20000,
                monitors=[logging_hook])
            
            # Configure the accuracy metric for evaluation
            metrics = {
                "accuracy":
                learn.MetricSpec(
                    metric_fn=tf.metrics.accuracy, prediction_key="classes"),
            }
            self.saver.save(session, self.savefile)
            # Evaluate the model and print results
            eval_results = syntactic_classifier.evaluate(
                x=eval_data, y=eval_labels, metrics=metrics)
            print(eval_results)  
            
def main():
    model = SyntacticClassifier("./model/model")
    model.fit()
    
    
if __name__ == '__main__':
    main()
        