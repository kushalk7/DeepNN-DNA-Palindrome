'''
python sticky_snippet_net.py mode model_file data_folder
'''

import os
import sys
import tensorflow as tf
import numpy as np
import string

#global variables
mode_fun = None
model_file = None
data_folder = None
dir_path = os.path.dirname(os.path.realpath(__file__))
count_nonstick = 0
count_12 = 0
count_34 = 0
count_56 = 0
count_78 = 0
count_stick = 0
learning_rate = 0.5
accuracy = None
epoch = 20

one_hot_dict = {
    "12-STICKY": 0,
    "34-STICKY": 1,
    "56-STICKY": 2,
    "78-STICKY": 3,
    "STICK_PALINDROME": 4,
    "NONSTICK": 5,
}

def matches(a, b):
    if (a == 'A' and b == 'C') or (a == 'C' and b == 'A'):
        return 1
    if (a == 'B' and b == 'D') or (a == 'D' and b == 'B'):
        return 1
    return 0


def get_label(dna):
    global count_12, count_34, count_56, count_78, count_nonstick, count_stick
    k = 0
    if not matches(dna[k],dna[-1]):
        count_nonstick += 1
        return "NONSTICK"

    while matches(dna[k],dna[-k-1]) and k < 20:
        k += 1

    if k == 1 or k == 2:
        count_12 += 1
        return "12-STICKY"
    if k == 3 or k == 4:
        count_34 += 1
        return "34-STICKY"
    if k == 5 or k == 6:
        count_56 += 1
        return "56-STICKY"
    if (k == 7 or k == 8) and k < 20 :
        count_78 += 1
        return "78-STICKY"
    count_stick += 1
    return "STICK_PALINDROME"


def one_hot_encoding(label):
    y_data = np.zeros((6,), dtype=np.int)
    y_data[one_hot_dict[label]] = 1
    return y_data


def translate(x):
    inp = "ABCD"
    out = "1234"
    s = string.translate(x, string.maketrans(inp,out))
    x_vec = [float(i) for i in s]
    # print "x_vec: ",x_vec
    return x_vec


def read_data():
    x = []
    y = []
    folder_path = os.path.join(dir_path, data_folder)
    if not os.path.isdir(folder_path):
        raise ValueError('FOLDER NOT FOUND')
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        raise ValueError('NO DATA')
    for f in files:
        filename = os.path.join(folder_path, f)
        # Check if file is a text file
        if filename.endswith('.txt'):
            filedata = open(filename,'r')
        else:
            continue

        for line in filedata:
            line = line.strip('\n')
            if len(line) != 40:
                continue
            # print line
            x_data = translate(line)
            # print "x_data: ",x_data
            # break
            y_data = one_hot_encoding(get_label(line))
            x.append(x_data)
            y.append(y_data)

    return np.transpose(x), np.transpose(y)


def perceptron(w_name, w_shape, input, b_name, b_shape, init):
    W = tf.get_variable(w_name, shape=w_shape, initializer=init)
    b = tf.get_variable(b_name, shape=b_shape, initializer=init)
    # print "x: ",input.shape, " W: ", w_shape, " b: ", b_shape
    return tf.nn.relu(tf.matmul(W, input) + b)


def nn():

    init = tf.contrib.layers.xavier_initializer()
    x = tf.placeholder(dtype=tf.float32, shape=(40, None))
    y = tf.placeholder(dtype=tf.float32, shape=(6, None))
    # layer 1
    y1 = perceptron("W1", (6, 40), x, "b1", (6, 1), init)

    # layer 2
    y2 = perceptron("W2", (6, 6), y1, "b2", (6, 1), init)

    # layer 3
    y3 = perceptron("W3", (6, 6), y2, "b3", (6, 1), init)

    # layer 4
    y4 = perceptron("W4", (6, 6), y3, "b4", (6, 1), init)

    # layer 5 (output)
    # W = tf.get_variable("W3", shape=(6, 10), initializer=init)
    # b = tf.get_variable("b3", shape=(10, 1), initializer=init)
    # y5 = tf.matmul(W, y4) + b

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y4))  ##reduce_sum

    global accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
    return train_step, x, y


def training():
    x_data, x_label = read_data()
    total = np.shape(x_data)[1]
    # print "total: ", total
    train, x, y = nn()
    # print "x: ", np.shape(x_data)
    # session = tf.Session()
    # session.run(tf.global_variables_initializer())
    # for i in range(0, total, 1000):
    #     end = i+1000
    #     batch = x_data[:, i:end]
    #     y_ = x_label[:, i:end]
    #     # print "batch: ", np.shape(batch), "y_: ", np.shape(y_)
    #     session.run(train,feed_dict={x: batch, y: y_}) #feed_dict=
    # saver = tf.train.Saver()
    # saver.save(session, model_file) #+".ckpt"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(epoch):
            for i in range(0, total, 100):
                end = i + 100
                batch = x_data[:, i:end]
                y_ = x_label[:, i:end]
                # print "i: ",i
                if i % 2 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch, y: y_})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train.run(feed_dict={x: batch, y: y_})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: x_data, y: x_label}))

        saver = tf.train.Saver()
        saver.save(sess, model_file)

    return 0


def fivefold_training():
    return 0


def testing():
    train, x, y = nn()
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, model_file)
    x_data, y_label = read_data()
    test_accuracy = accuracy.eval(session=session,feed_dict={x: x_data, y: y_label})
    print('Testing accuracy %g' % (test_accuracy))
    return 0


mode = {
    "train": training,
    "5fold": fivefold_training,
    "test": testing
}


def print_help():
    print "python sticky_snippet_net.py mode model_file data_folder"


def parse_args():
    print sys.argv
    global mode_fun, model_file, data_folder
    mode_fun = mode[sys.argv[1]]
    model_file = str(sys.argv[2])
    data_folder = str(sys.argv[3])


if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print "Invalid arguments"
        print_help()
    else:
        parse_args()
        # read_data()
        mode_fun()
