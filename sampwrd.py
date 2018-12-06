 here is how i did it, the data is from sentdex's website used in previous tutorials:

from nltk.tokenize import word_tokenize
import tensorflow as tf
from nltk.corpus import stopwords
import random
import nltk
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from sklearn import cross_validation

positive_file = open("F:\Artificial Intelligence\\tensorflow\\positive.txt", 'r').read()
negative_file = open("F:\Artificial Intelligence\\tensorflow\\negative.txt", 'r').read()

documents = []

for r in positive_file.split('\n'):
    documents.append([r, 1, 0])

for r in negative_file.split('\n'):
    documents.append([r, 0, 1])

doc = np.array(documents)

all_words = []
positive_words = word_tokenize(positive_file)
negative_words = word_tokenize(negative_file)

for w in positive_words:
    all_words.append(w.lower())
for w in negative_words:
    all_words.append(w.lower())

all_word = nltk.FreqDist(all_words)

word_features = list(all_word.keys())[:10000]


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:]:
            current_words = word_tokenize(l.lower())

            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset


fet = []

fet += sample_handling("F:\Artificial Intelligence\\tensorflow\\positive.txt", word_features, [1, 0])
fet += sample_handling("F:\Artificial Intelligence\\tensorflow\\negative.txt", word_features, [0, 1])
random.shuffle(fet)
newfeatures = np.array(fet)

testing_size = int(0.3 * len(newfeatures))
train_x = list(newfeatures[:, 0][:-testing_size])
train_y = list(newfeatures[:, 1][:-testing_size])

test_x = list(newfeatures[:, 0][-testing_size:])
test_y = list(newfeatures[:, 1][-testing_size:])

train_x = np.array(train_x, dtype='int')
train_y = np.array(train_y, dtype='int')
test_x = np.array(test_x, dtype='int')
test_y = np.array(test_y, dtype='int')
print(len(train_x))
hm_epochs = 50
nmclasses = 2
batch_size = 933
chunk_size = 100
n_chunks = 100
rnn_size = 512

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrentNetwork(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, nmclasses])),
              'biases': tf.Variable(tf.random_normal([nmclasses]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def trainNeuralNetwork(x):
    prediction = recurrentNetwork(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_cost = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]

                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))

                batch_y = train_y[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_cost += c
                i += batch_size
            print("Epoch ", epoch + 1, 'completed out of ', hm_epochs, 'cost ', epoch_cost)

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x: test_x.reshape((-1, n_chunks, chunk_size)), y: test_y}))


trainNeuralNetwork(x)﻿ 