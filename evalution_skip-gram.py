from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from training_set_builder import SampleBuilder
from evaluate_cnn import convert_from_urls
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

    
    #pool1 = tf.layers.max_pooling2d(inputs=tf.add(tf.add(conv1, conv2), tf.add(conv2, conv4)), pool_size=[32, 64], strides=1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[16, 64], strides=2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[16, 64], strides=2)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[16, 64], strides=2)
    #pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[16, 64], strides=1)
    #pool5 = tf.layers.max_pooling2d(inputs=[pool1 + pool2 + pool3 + pool4], pool_size=[16, 64], strides=1) 
    #For now, 4/23/17, I will use only one conv and pooling layer
    # Convolutional Layer #2 and Pooling Layer #2
    """
    conv4 = tf.layers.conv2d(
      inputs = pool1,
      filters = 20,
      kernel_size = [2, 32],
      padding = "same",
      activation = tf.nn.relu) 
    
    conv5 = tf.layers.conv2d(
          inputs = pool2,
          filters = 20,
          kernel_size = [2, 32],
          padding = "same",
          activation = tf.nn.relu)    
    conv6 = tf.layers.conv2d(
          inputs = pool3,
          filters = 20,
          kernel_size = [2, 32],
          padding = "same",
          activation = tf.nn.relu)    
          
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
    """
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

    evaluate()
    

def evaluate():
    # Load training and eval data
    builder = SampleBuilder(['difficult','easy'])
    #data, labels = builder.extract_samples_from_pdf('/Users/David/Desktop/Dev/capstone/syntactic-classifier/lib/pdfs/difficult/kant.pdf', 'difficult')
    #data, labels = builder.extract_samples_from_pdf('/Users/David/Downloads/Wittgenstein-Tractatus.pdf', 'difficult')
    #data, labels = builder.extract_samples_from_pdf('/Users/David/Downloads/tuck-everlasting-bookfiles.pdf', 'easy')
    
    #data, labels = builder.load_dataset_from_disk()

    urls = [('https://en.wikipedia.org/wiki/Gentrification', 0),
            ('https://plato.stanford.edu/entries/perceptual-learning/', 0), 
            ('https://plato.stanford.edu/entries/logical-empiricism/', 0),
            ('https://en.wikipedia.org/wiki/Quantum_mechanics', 0),
            ('http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/', 0),
            ('http://mcb.asm.org/content/23/11/3721.full', 0),
            ('http://web.math.rochester.edu/people/faculty/jnei/algtop.html', 0),
            ('http://web.stanford.edu/dept/HPS/critstudies/sunny.html', 0),
            ('http://csmt.uchicago.edu/glossary2004/symbolicrealimaginary.htm', 0),
            ('http://brucelevine.net/liberation-psychology/', 0),
            ('https://simple.wikipedia.org/wiki/Government_of_Australia', 1),
            ('https://www.quora.com/Can-you-explain-Darwins-theory-of-evolution-in-simple-language-and-why-some-people-reject-it', 1),
            ('https://www.mathsisfun.com/algebra/introduction.html', 1),
            ('http://www.localhistories.org/america.html', 1),
            ('http://www.explainthatstuff.com/howcomputerswork.html', 1),
            ('https://simple.wikipedia.org/wiki/Ecology', 1),
            ('http://www.thwink.org/sustain/glossary/Sustainability.htm', 1),
            ('http://www.space.com/16014-astronomy.html', 1),
            ('https://www.simplypsychology.org/psychoanalysis.html', 1),
            ('http://science.howstuffworks.com/science-vs-myth/everyday-myths/anthropic-principle.htm', 1)]

    #urls = [('https://en.wikipedia.org/wiki/Gentrification', 0)]    
    data, labels = convert_from_urls(urls)
    eval_data = np.array(data, dtype=np.float32)
    eval_labels = np.array(labels, dtype=np.int32)
    train_data = np.array(data, dtype = np.float32)
    train_labels = np.array(labels, dtype=np.int32)

    print("Loaded training and eval data.")


    #EFFECTIVE MODELS?
    #26
    #27 appears to be similar to 26
    #35 has 2,3,4, kernel sizes
    #60works ish
    #61 seems good
    #72 VERY PROMISING. ended up okay. 2 conv stages, LR=.1 DR=.1
    #74 also promising, one conv stage, lr=.01
    
    syntactic_classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="./model/syntactic-classifier_convnet_model82")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
            
    print("Set up logging.")
      

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
    
#below is model 26
"""
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

    
    #pool1 = tf.layers.max_pooling2d(inputs=tf.add(tf.add(conv1, conv2), tf.add(conv2, conv4)), pool_size=[32, 64], strides=1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    #pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[16, 64], strides=1)
    #pool5 = tf.layers.max_pooling2d(inputs=[pool1 + pool2 + pool3 + pool4], pool_size=[16, 64], strides=1) 
    #For now, 4/23/17, I will use only one conv and pooling layer
    # Convolutional Layer #2 and Pooling Layer #2
    
    conv4 = tf.layers.conv2d(
      inputs = pool1,
      filters = 20,
      kernel_size = [2, 32],
      padding = "same",
      activation = tf.nn.relu) 
    
    conv5 = tf.layers.conv2d(
          inputs = pool2,
          filters = 20,
          kernel_size = [2, 32],
          padding = "same",
          activation = tf.nn.relu)    
    conv6 = tf.layers.conv2d(
          inputs = pool3,
          filters = 20,
          kernel_size = [2, 32],
          padding = "same",
          activation = tf.nn.relu)    

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
    # Dense Layer
    pool1_flat = tf.reshape(pool4, [-1, 4 * 16 * 20])
    pool2_flat = tf.reshape(pool5, [-1, 4 * 16 * 20])
    pool3_flat = tf.reshape(pool6, [-1, 4 * 16 * 20])
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
            learning_rate=0.01,
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
        """