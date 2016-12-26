# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import os

data_dir = '/home/jsemko/data/udacity'

pickle_file = 'notMNIST.pickle'
pickle_file = os.path.join(data_dir, pickle_file)


with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    del save  # hint to help gc free up memory

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
  	dataset = dataset.reshape(
	   	(-1, image_size, image_size, num_channels)).astype(np.float32)
  	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels, return_wrong=False):
    correct = np.argmax(predictions, 1) == np.argmax(labels, 1)
    acc=  (100.0 * np.sum(correct)) / predictions.shape[0]
    if return_wrong:
        return acc, ~correct
    else:
        return acc


batch_size = 128
patch_size_1 = 5
patch_size_2 = 3
depth_1 = 16
depth_2 = 64
num_hidden = 1024

beta = .002
graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32,
        shape=(batch_size, image_size, image_size, num_channels)
    )

    tf_train_labels = tf.placeholder(
		tf.float32,
        shape=(batch_size, num_labels)
    )
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.Variable(
        tf.truncated_normal(
            [patch_size_1, patch_size_1, num_channels, depth_1],
            stddev=0.1
        )
    )
    layer1_biases = tf.Variable(tf.zeros([depth_1]))

    layer2_weights = tf.Variable(
        tf.truncated_normal(
            [patch_size_2, patch_size_2, depth_1, depth_2],
            stddev=0.1
        )
    )
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_2]))

    layer3_weights = tf.Variable(
        tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth_2, num_hidden],
            stddev=0.1
        )
    )
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(
        tf.truncated_normal(
            [num_hidden, num_labels],
            stddev=0.1
        )
    )
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    tf_beta = tf.constant(beta)
    # Model.
    def model(data, keep_prob=.9, keep_prob_p=1):
        conv = tf.nn.conv2d(
            tf.nn.dropout(data, keep_prob),
            layer1_weights,
            [1, 1, 1, 1],
            padding='SAME'
        )
        hidden = tf.nn.dropout(
            tf.nn.relu(conv + layer1_biases),
            keep_prob_p
        )

        pooled = tf.nn.max_pool(
            hidden,
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            padding='SAME',
        )
        pooled = tf.nn.dropout(
            pooled,
            keep_prob
        )
        conv = tf.nn.conv2d(
            pooled,
            layer2_weights,
            [1, 1, 1, 1],
            padding='SAME'
        )
        hidden = tf.nn.dropout(
            tf.nn.relu(conv + layer2_biases),
            keep_prob_p
        )

        pooled = tf.nn.max_pool(
            hidden,
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            padding='SAME',
        )
        pooled = tf.nn.dropout(
            pooled,
            keep_prob
        )
        shape = pooled.get_shape().as_list()
        reshape = tf.reshape(
            pooled,
            [shape[0], shape[1] * shape[2] * shape[3]]
        )

        hidden = tf.nn.dropout(
            tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases),
            keep_prob
        )

        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        ) + tf_beta * (
        tf.nn.l2_loss(layer1_biases) +
        tf.nn.l2_loss(layer2_biases) +
        tf.nn.l2_loss(layer3_biases) +
        tf.nn.l2_loss(layer4_biases) +
        tf.nn.l2_loss(layer1_weights) +
        tf.nn.l2_loss(layer2_weights) +
        tf.nn.l2_loss(layer3_weights) +
        tf.nn.l2_loss(layer4_weights)
    )
    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.04, global_step, 100, .95)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )
    #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, keep_prob=1))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, keep_prob=1))

num_steps = 2001

with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        _, l, predictions = session.run(
            [optimizer, loss, train_prediction],
            feed_dict=feed_dict
        )

        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
