# basic imports for arrays and plot
import numpy as np
from matplotlib import pyplot as plt
# import self-organised map implementation
from minisom import MiniSom


def test_minisom():
    """
    Test if minisom is installed correctly and works
    as expected

    :return: None
    """
    # test parameters
    dim = 20
    iterations = 250

    rgb_random_data = np.random.uniform(0, 1, (dim * dim, 3))

    som = MiniSom(dim, dim, 3, sigma=1.0, learning_rate=5)
    som.random_weights_init(rgb_random_data)
    print("Training...")
    som.train_random(rgb_random_data, iterations)
    print("...ready!")
    print("Shape is " + str(som.get_weights().shape))
    print(som.get_weights())
    output_rgb = plt.imshow(som.get_weights())

    # Show figure
    plt.figure(1)

    # original subplot
    plt.subplot(121)
    plt.title("Original array")
    plt.imshow(rgb_random_data)

    # SOM result subplot
    plt.subplot(122)
    plt.title("SOM result")
    plt.imshow(output_rgb)

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#    test_minisom()
