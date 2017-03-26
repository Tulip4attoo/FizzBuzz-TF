# You can see the original version and idea here:
# http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import tensorflow as tf
import numpy as np
import time

###### PART 1: PREPARE FIXED VARs AND FUNCTIONs ######
######################################################
######################################################

NUM_DIGITS = 13
NUM_LABELS = 4
X_WIDTH = NUM_DIGITS
NUM_NODES = 128
BATCH_SIZE = 128
GRADIENT_CONSTANT = 0.05
NUM_STEPS = 120001

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
  return np.array([i >> d & 1 for d in range(num_digits)])[::-1]

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
  if   i % 15 == 0: return np.array([0, 0, 0, 1])
  elif i % 5  == 0: return np.array([0, 0, 1, 0])
  elif i % 3  == 0: return np.array([0, 1, 0, 0])
  else:             return np.array([1, 0, 0, 0])

# Calculate the accuracy of the model
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz(i, prediction):
  return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])
trX = trX.astype(np.float32)
trY = trY.astype(np.float32)

teX = np.array([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
trY = np.array([fizz_buzz_encode(i)          for i in range(1, 101)])
teX = teX.astype(np.float32)
teY = teY.astype(np.float32)



######  PART 2: CREATE A NEURAL NETWORK MODEL   ######
######################################################
######################################################

graph = tf.Graph()
with graph.as_default():

  tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, X_WIDTH))
  tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
  tf_test_dataset = tf.constant(teX)

  weights_1 = tf.Variable(
    tf.truncated_normal([X_WIDTH, NUM_NODES]))
  biases_1 = tf.Variable(tf.zeros([NUM_NODES]))
  
  tf_hidden_nodes = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)

  weights_2 = tf.Variable(
    tf.truncated_normal([NUM_NODES, NUM_LABELS]))
  biases_2 = tf.Variable(tf.zeros([NUM_LABELS]))

  logits = tf.matmul(tf_hidden_nodes, weights_2) + biases_2

  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  optimizer = tf.train.GradientDescentOptimizer(GRADIENT_CONSTANT).minimize(loss)

  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(tf.matmul(
    tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)
  tf.add_to_collection('params', weights_1)
  tf.add_to_collection('params', biases_1)
  tf.add_to_collection('params', weights_2)
  tf.add_to_collection('params', biases_2)
  saver = tf.train.Saver()



######      PART 3: RUN AND LOAD THE MODEL      ######
######################################################
######################################################

with tf.Session(graph=graph) as session:
  start_time = time.time()
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(NUM_STEPS):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    offset = np.random.randint(0, len(trX) - 1, BATCH_SIZE)
    batch_data = trX[offset]
    batch_labels = trY[offset]

    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
    if (step % 10000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      saver.save(session, 'my-model', global_step=step)
  # Calling .eval() on valid_prediction is basically like calling run(), but
  # just to get that one numpy array. Note that it recomputes all its graph
  # dependencies.
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print("--- %s seconds ---" % (time.time() - start_time))



# Load model part. TBH it is not necessery
with tf.Session(graph=graph) as sess:
  new_saver = tf.train.import_meta_graph('my-model-120000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  all_vars = tf.get_collection('params')
  weights_1, biases_1, weights_2, biases_2 = all_vars[0], all_vars[1], \
                                                all_vars[2], all_vars[3]
  tf_test_dataset = tf.constant(teX)
  test_prediction = tf.nn.softmax(tf.matmul(
    tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2)
  predict_op = tf.argmax(test_prediction, 1)
  teY, tf_onehot_output = sess.run([predict_op, test_prediction])
  output = np.vectorize(fizz_buzz)(numbers, teY)











