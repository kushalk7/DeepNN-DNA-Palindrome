'''
python sticky_snippet_net.py mode model_file data_folder
'''

import os
import sys
import tensorflow as tf
import numpy as np
import string
import timeit

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
learning_rate = 0.01
accuracy = None
loss_fun = None
epoch = 20
batch_size = 1000
num_classes = 6
confusion = None

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
    out = "0123"
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

    return x,y #np.transpose(x), np.transpose(y)

# def print_number()

def perceptron(w_name, w_shape, input, b_name, b_shape, init):
    W = tf.get_variable(w_name, shape=w_shape, initializer=init)
    b = tf.get_variable(b_name, shape=b_shape, initializer=init)
    # print "x: ",input.shape, " W: ", w_shape, " b: ", b_shape
    return tf.nn.relu(tf.matmul(input, W) + b)


def nn(confusion_matrix = False):

    init = tf.contrib.layers.xavier_initializer()
    x = tf.placeholder(dtype=tf.float32, shape=(None, 40))
    y = tf.placeholder(dtype=tf.float32, shape=(None, 6))
    # layer 1
    y1 = perceptron("W1", (40, 6), x, "b1", (1, 6), init)

    # layer 2
    y2 = perceptron("W2", (6, 20), y1, "b2", (1, 20), init)

    # layer 3
    y3 = perceptron("W3", (20, 40), y2, "b3", (1, 40), init)

    # layer 4
    y4 = perceptron("W4", (40, 6), y3, "b4", (1, 6), init)

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y4))  ##reduce_sum

    global accuracy,loss_fun
    loss_fun = cross_entropy_loss
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    if(confusion_matrix):
        global confusion
        confusion = tf.confusion_matrix(labels=tf.argmax(y, 1), predictions=tf.argmax(y4, 1), num_classes=num_classes)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
    return train_step, x, y  #  y4, x,y#


def training():
    x_data, x_label = read_data()
    total = np.shape(x_data)[0]
    # print "total: ", total
    train, x, y = nn()
    global loss_fun
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cnt = 0
        for e in range(epoch):
            start = timeit.default_timer()
            for i in range(0, total, batch_size):
                end = i + batch_size
                batch = x_data[i:end]
                y_ = x_label[i:end]
                cnt += batch_size
                train.run(feed_dict={x: batch, y: y_})

                if cnt % 1000 == 0:
                    print cnt," items trained"

            stop = timeit.default_timer()

            print "Total testing time for epoch: ", e, " =>", stop - start, " seconds"
                # sess.run([train_step, check_op], feed_dict={x: batch, y: y_})
                # if i % 10 == 0:
                #     train_accuracy = accuracy.eval(feed_dict={x: batch, y: y_})
                #     print('step %d, training accuracy %g' % (i, train_accuracy))
                    # loss = loss_fun.eval(feed_dict={x: batch, y: y_})
                    # print "####step: ", i, "loss: ", loss

        # print "####EPOCH: ", e,"loss: ",loss
        # print('test accuracy %g' % accuracy.eval(feed_dict={x: x_data, y: x_label}))
        print "Processing complete!"
        print "Total number items trained on : ", total

        saver = tf.train.Saver()
        saver.save(sess, model_file)

    return 0


def fivefold_training():
    x_data, y_label = read_data()
    total = np.shape(x_data)[0]
    fold = total/5
    set_x = [[] for _ in xrange(5)]
    set_y = [[] for _ in xrange(5)]
    j = 0
    # print "total: ", total

    # Divide data into 5 equal sets
    for i in range(0, total, fold):
        end = i + fold
        if end > total:
            end = total
        # print "i: ", i, " end: ", end
        set_x[j] = x_data[i:end]
        set_y[j] = y_label[i:end]
        j += 1

    train, x, y = nn(True)
    total_acc = 0

    # Iteratively choose each set from testing and rest for training
    for i in range(5):
        train_data, train_label = [], []
        test_data, test_label = [], []
        for j in range(5):
            if i != j:
                if len(train_data) == 0:
                    train_data = set_x[i]
                    train_label = set_y[i]
                else:
                    train_data = np.append(train_data, set_x[i], axis=0)
                    train_label = np.append(train_label, set_y[i], axis=0)
            else:
                if len(test_data) == 0:
                    test_data = set_x[i]
                    test_label = set_y[i]
                else:
                    test_data = np.append(test_data, set_x[i], axis=0)
                    test_label = np.append(test_label, set_y[i], axis=0)
        # print "train_data: ", np.shape(train_data), " test: ", np.shape(test_data)
        print

        total = train_data.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epoch):
                cnt = 0
                print "Epoch: ", e, " starting"
                for r in range(0, total, batch_size):
                    end = r + batch_size
                    batch = train_data[r:end]
                    y_ = train_label[r:end]
                    # print "i: ",i
                    cnt += batch_size
                    if cnt % 1000 == 0:
                        print cnt ," items trained"
                        # train_accuracy = accuracy.eval(feed_dict={x: batch, y: y_})
                        # print('step %d, training accuracy %g' % (r, train_accuracy))

                    start = timeit.default_timer()

                    train.run(feed_dict={x: batch, y: y_})

                    stop = timeit.default_timer()

                    print "Total training time for epoch: ",e," =>",stop - start, " seconds"


            # print('training accuracy(entire train data) %g' % accuracy.eval(feed_dict={x: train_data, y: train_label}))

            #testing
            start = timeit.default_timer()

            acc = accuracy.eval(feed_dict={x: test_data, y: test_label})

            stop = timeit.default_timer()

            print "Total testing time for epoch: ", e, " =>", stop - start, " seconds"

            # print('testing accuracy(entire test data) %g' % acc)
            print "Processing complete!"
            print "Total number items trained on : ", np.shape(train_data)[0]
            print "Total number items tested on : ", np.shape(test_data)[0]
            total_acc += acc

            global confusion
            print "Confusion Matrix: "
            print tf.Tensor.eval(confusion,feed_dict={x: test_data, y: test_label}, session=sess)

            if i == 4 :
                saver = tf.train.Saver()
                saver.save(sess, model_file)

    print "Aggregate Accuracy: ", total_acc/5.0

    return 0


def testing():
    train, x, y = nn()
    session = tf.Session()
    saver = tf.train.Saver()
    saver.restore(session, model_file)
    x_data, y_label = read_data()
    # print "x_data: ",x_data.shape, "y_label: ", y_label.shape
    test_accuracy = accuracy.eval(session=session,feed_dict={x: x_data, y: y_label})

    print "Processing complete!"
    print "Total number items tested on : ", np.shape(x_data)[0]
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
