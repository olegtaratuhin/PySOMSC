# file contains testing som on rgb data
import numpy as np
from matplotlib import pyplot as plt

from tf_som_implementation import Som


# Test SOM Network
def test(xdim=20, ydim=10, iterations=100):
    """
    Testing on rgb generated randomly by numpy

    :param xdim: X-dimension of som
    :param ydim: Y-dimension of som
    :param iterations: number of iterations
    :return: None
    """
    # Training inputs for RGB colors
    colors = np.random.uniform(0, 1, (xdim * ydim, 3))
    print("Test generated")

    som = Som(xdim, ydim, 3, iterations)
    print("SOM created")

    print("SOM training ...")
    som.train(colors)
    print("SOM training done")

    # Get output grid of som
    image_grid = som.get_centroids()

    plt.figure(1)

    plt.subplot(121)
    plt.title("Original random colors")

    colors_2d = np.zeros((xdim, ydim, 3))
    # rearrange colors to a (xdim, ydim, 3) shape in order
    # to be displayed correctly on a grid
    for x in range(xdim):
        for y in range(ydim):
            colors_2d[x][y] = colors[x + xdim * y]
    plt.imshow(colors_2d)

    plt.subplot(122)
    plt.title("SOM processed colors")
    plt.imshow(image_grid)
    plt.show()


if __name__ == "__main__":
    test()
