# file contains testing som on mnist data
import random

import numpy as np
from matplotlib import pyplot as plt
# test on provided by tensorflow mnist data
from tensorflow.examples.tutorials.mnist import input_data

from tf_som_implementation import Som

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def test(size_train=100, size_test=110, iterations=200):
    """
    Test som on mnist data provided by tensorflow
    :param size_train:
    :param size_test:
    :param iterations:
    :return:
    """

    if size_test < size_train:
        size_test = int(1.1 * size_train)

    # get train vectors
    x_train, y_train = mnist.train.images[:size_train, :],\
                       mnist.train.labels[:size_train, :]

    # get test vectors and trim accordingly with sizes
    x_test, y_test = mnist.train.images[:size_test, :], \
                     mnist.train.labels[:size_test, :]
    x_test, y_test = x_test[size_train:size_test, :], \
                     y_test[size_train:size_test, :]
    print("Set training vectors")

    # train som
    som = Som(30, 30, x_train.shape[1], iterations)
    print("Som created")

    som.train(x_train)
    print("Training completed")
    print("Parsing results")

    # results on training vector
    mapped_train = np.array(som.map_vects(x_train))
    x1, y1 = mapped_train[:, 0], mapped_train[:, 1]

    index_train = [np.where(r == 1)[0][0] for r in y_train]
    index_train = list(map(str, index_train))

    # results on testing vector
    mapped_test = np.array(som.map_vects(x_test))
    x2, y2 = mapped_test[:, 0], mapped_test[:, 1]

    index_test = [np.where(r == 1)[0][0] for r in y_test]
    index_test = list(map(str, index_test))

    def add_text(index_cur, mapped_cur, color="white"):
        """
        Add text to plot
        :return:
        """
        for i, m in enumerate(mapped_cur):
            plt.text(m[0], m[1], index_cur[i], ha='center', va='center',
                     bbox=dict(facecolor=color, alpha=0.5, lw=0))

    def display_digit(num):
        """
        function to draw a table with numbers
        :param num:
        :return:
        """
        label = y_train[num].argmax(axis=0)
        image = x_train[num].reshape([28, 28])
        plt.title('Example: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()

    display_digit(random.randint(0, x_train.shape[0]))

    # 2 subplots: training and testing + training
    plt.figure(1)

    # training
    plt.subplot(121)
    plt.scatter(x1, y1)
    add_text(index_train, mapped_train)
    plt.title("MNIST training size=" + str(size_train))

    plt.subplot(122)
    # results from training
    plt.scatter(x1, y1)
    add_text(index_train, mapped_train)
    # results from testing
    plt.scatter(x2, y2)
    add_text(index_test, mapped_test, color="red")
    plt.title("MNIST testing size=" + str(size_test) +
              " upon training size=" + str(size_train))

    plt.show()


if __name__ == '__main__':
    test()
