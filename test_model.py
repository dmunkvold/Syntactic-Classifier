import tensorflow as tf


with tf.Session() as session:
    b = tf.Variable(0.0, name="bias")
    saver = tf.train.Saver()
    saver.restore(session, './model/syntactic-classifier_convnet_model/graph.pbtxt')
    metrics = {
              "accuracy":
                  learn.MetricSpec(
                      metric_fn=tf.metrics.accuracy, prediction_key="classes"),
          }
        
    # Evaluate the model and print results
    eval_results = syntactic_classifier.evaluate(
    x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)      