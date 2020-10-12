import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
size_batches = 100
 
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

	hl1 = {'W': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'B': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hl2 = {'W': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'B': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hl3 = {'W': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'B': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	ol = {'W': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'B': tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hl1['W']), hl1['B'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hl2['W']), hl2['B'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hl3['W']), hl3['B'])
	l3 = tf.nn.relu(l3)

	ol = tf.matmul(l3, ol['W']), ol['B']

	return output

def train_nn(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.sofmax_cross_entropy_with_logits(prediction, y)), 


