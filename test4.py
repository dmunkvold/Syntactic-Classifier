import tensorflow as tf






v1 = tf.Variable(0,0, name="v1")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(init_op)
  saver.restore(sess, "./model/syntactic-classifier_convnet_model/model.ckpt-36600.index")
  