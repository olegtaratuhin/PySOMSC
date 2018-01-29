# basic imports for arrays and plot
import numpy
# import self-organised map implementation
from minisom import MiniSom
from pylab import imshow, show


def test_minisom():
    """
    Test if minisom is installed correctly and works
    as expected

    :return: None
    """
    # test parameters
    dim = 20
    iterations = 2500

    # make sure the data is generated correctly
    rgb_random_data = numpy.random.uniform(0, 1, (iterations, 3))
    imshow(rgb_random_data)
    show()

    som = MiniSom(dim, dim, 3, sigma=dim/4, learning_rate=0.5)
    print("Training...")
    som.train_random(rgb_random_data, iterations)
    print("...ready!")

    imshow(som.get_weights())
    show()


test_minisom()
