import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 10))

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)


# training data
mnist = fetch_mldata("MNIST original")
x = mnist.data.astype(np.float32)
x /= np.max(x, axis=1, keepdims=True)
y = mnist.target.astype(np.int32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)

model = MLP()

optimizer = chainer.optimizers.SGD(lr=0.05)
optimizer.use_cleargrads()
optimizer.setup(model)

for i in range(1, 11):
    for j in range(0, len(x_train), 100):
        model.cleargrads()
        logit = model(x_train[j: j + 100])
        loss = F.softmax_cross_entropy(logit, y_train[j: j + 100])
        loss.backward()
        optimizer.update()
    accuracy = F.accuracy(model(x_test), y_test)
    print("epoch {0:02d}, accuracy {1:.02f}".format(i, accuracy.data.item()))
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
