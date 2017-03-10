from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


DATA_DIR = '/media/jeremy/San/Data/udacity_data/deeplearning'

PICKLE_FILE = 'notMNIST.pickle'
PICKLE_FILE = os.path.join(DATA_DIR, PICKLE_FILE)

IMAGE_SIZE = 28
NUM_LABELS = 10
BATCH_SIZE = 128
HL_SIZE = 512

def main():
    """Train and test a feed forward network"""

    with open(PICKLE_FILE, 'rb') as f_stream:
        save = pickle.load(f_stream)
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


    def reformat(dataset, labels):

        """Puts data into one-hot encoding"""

        dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)

        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)

        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 10000

    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(
            tf.truncated_normal(
                [IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]
            )
        )
        biases = tf.Variable(tf.zeros([NUM_LABELS]))

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf_train_labels)
        )

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases
        )

        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases
        )

    num_steps = 801

    def accuracy(predictions, labels):
        """Gives the accuracy of classifier"""
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases.
        tf.global_variables_initializer().run()
        print('Initialized')

        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, mb_loss, predictions = session.run([optimizer, loss, train_prediction])
            if step % 100 == 0:
                print('Loss at step {:d}: {:f}'.format(step, mb_loss))
                print(
                    'Training accuracy: %.1f%%' % accuracy(
                        predictions, train_labels[:train_subset, :]
                    )
                )
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print(
                    'Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), valid_labels
                    )
                )

        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE)
        )
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, NUM_LABELS)
        )
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS])
        )
        biases = tf.Variable(tf.zeros([NUM_LABELS]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf_train_labels)
        )

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(tf_test_dataset, weights) + biases
        )

    num_steps = 10001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):

            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {
                tf_train_dataset : batch_data,
                tf_train_labels : batch_labels
            }
            _, mb_loss, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict
            )

            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, mb_loss))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print(
                    "Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), valid_labels
                    )
                )
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


    # Main Problem: Use above template to make 1 layer NN


    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE)
        )
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(BATCH_SIZE, NUM_LABELS)
        )
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = tf.Variable(
            tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, HL_SIZE])
        )
        biases_1 = tf.Variable(tf.zeros([HL_SIZE]))

        weights_2 = tf.Variable(
            tf.truncated_normal([HL_SIZE, NUM_LABELS])
        )
        biases_2 = tf.Variable(tf.zeros([NUM_LABELS]))


        # Training computation.
        neurons = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)

        logits = tf.matmul(neurons, weights_2) + biases_2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf_train_labels
            )
        )

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu(
                    tf.matmul(tf_valid_dataset, weights_1) + biases_1
                ),
                weights_2
            ) + biases_2
        )
        test_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu(
                    tf.matmul(tf_test_dataset, weights_1) + biases_1
                ),
                weights_2
            ) + biases_2
        )

    num_steps = 10001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):

            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {
                tf_train_dataset : batch_data,
                tf_train_labels : batch_labels
            }
            _, mb_loss, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict
            )

            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, mb_loss))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels)
                     )
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

if __name__ == '__main__':
    main()
