import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

# training data
mnist = fetch_mldata("MNIST original")
x = mnist.data.astype(np.float32)
x /= np.max(x, axis=1, keepdims=True)
y = mnist.target.astype(np.int32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)

model = L.Linear(784, 10)

optimizer = chainer.optimizers.SGD()
optimizer.use_cleargrads()
optimizer.setup(model)

for i in range(1, 101):
    for j in range(0, len(x_train), 100):
        model.cleargrads()
        logit = model(x_train[j: j + 100])
        loss = F.softmax_cross_entropy(
            logit, y_train[j: j + 100])
        loss.backward()
        optimizer.update()
    indices = np.random.choice(10000, 100, replace=False)
    accuracy = F.accuracy(model(x_test), y_test).data.item()
    print("step {0:03d}, accuracy {1:.02f}".format(i, accuracy))
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
