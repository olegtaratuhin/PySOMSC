import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Testing somclu and components
def test_somclu():
    c1 = np.random.rand(50, 3) / 5
    c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3) / 5
    c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3) / 5
    data = np.float32(np.concatenate((c1, c2, c3)))
    colors = ["red"] * 50
    colors.extend(["green"] * 50)
    colors.extend(["blue"] * 50)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
    # labels = range(150)


# if __name__ == '__main__':
print("Hello, SOMSC implementation")
test_somclu()
