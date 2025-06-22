# for plotting decision boundary
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# neccesary nn functions
from Dense import Dense
from mse import mse, mse_prime
from activations import Tanh
import numpy as np
from network import train, predict


X = np.reshape([[0,0], [0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4,1,1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh(),
]

train(network, mse, mse_prime, X, Y, epoch = 1000, learning_rate=0.01, verbose=True)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()