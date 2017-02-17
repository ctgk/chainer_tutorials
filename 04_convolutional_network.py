import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", "-g", default=-1, type=int,
    help="negative value indicates no gpu, default=-1")
args = parser.parse_args()


class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5, pad=2),
            conv2=L.Convolution2D(32, 64, 5, pad=2),
            fc3=L.Linear(7 * 7 * 64, 1000),
            fc4=L.Linear(1000, 10))

    def __call__(self, x, train=False):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc3(h)), ratio=0.5, train=train)
        return self.fc4(h)


# training data
mnist = fetch_mldata("MNIST original")
x = np.float32(mnist.data)
x /= np.max(x, axis=1, keepdims=True)
x = x.reshape(-1, 1, 28, 28)
y = np.int32(mnist.target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)

model = CNN()
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = chainer.cuda.cupy if args.gpu >= 0 else np
x_train = xp.asarray(x_train)
x_test = xp.asarray(x_test)
y_train = xp.asarray(y_train)
y_test = xp.asarray(y_test)

optimizer = chainer.optimizers.Adam(alpha=1e-4)
optimizer.use_cleargrads()
optimizer.setup(model)

for i in range(1, 11):
    for j in range(0, len(x_train), 50):
        model.cleargrads()
        logit = model(x_train[j: j + 50], train=True)
        loss = F.softmax_cross_entropy(logit, y_train[j: j + 50])
        loss.backward()
        optimizer.update()
    accuracy = F.accuracy(model(x_test), y_test)
    print("step {0:02d}, accuracy {1}".format(i, accuracy.data))
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
