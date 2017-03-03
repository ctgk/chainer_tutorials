import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata


def xavier_initializer(size):
    return chainer.initializers.Normal(scale=1 / size)


class MLP(chainer.Chain):

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_hid, initialW=xavier_initializer(n_in)),
            l2=L.Linear(n_hid, n_out, initialW=xavier_initializer(n_hid)))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return F.clip(F.sigmoid(self.l2(h)), 1e-6, 1 - 1e-6)


mnist = fetch_mldata("MNIST original")
x = np.float32(mnist.data)
x /= np.max(x, axis=1, keepdims=True)

G = MLP(100, 400, 28 * 28)
D = MLP(28 * 28, 400, 1)

g_optimizer = chainer.optimizers.Adam()
g_optimizer.use_cleargrads()
g_optimizer.setup(G)
d_optimizer = chainer.optimizers.Adam()
d_optimizer.use_cleargrads()
d_optimizer.setup(D)

for i in range(1, 101):
    for j in range(0, len(x), 100):
        G.cleargrads()
        D.cleargrads()
        z = np.float32(np.random.normal(size=(100, 100)))
        g_loss = -F.sum(F.log(D(G(z))))
        g_loss.backward()
        g_optimizer.update()

        G.cleargrads()
        D.cleargrads()
        z = np.float32(np.random.normal(size=(100, 100)))
        d_loss = -F.sum(F.log(D(x[j: j + 100])) + F.log(1 - D(G(z))))
        d_loss.backward()
        d_optimizer.update()
    print("epoch {0:03d}, g_loss {1}, d_loss {2}".format(i, g_loss.data, d_loss.data))
    images = G(np.float32(np.random.normal(size=(25, 100)))).data
    for index, img in enumerate(images):
        plt.subplot(5, 5, index + 1)
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.savefig("result{0:02d}.png".format(i))
    plt.clf()
