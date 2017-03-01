import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, report, training
from chainer.training import extensions
import numpy as np
from sklearn.datasets import fetch_mldata


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 10))

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)


class Classifier(chainer.Chain):

    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


# training data
mnist = fetch_mldata("MNIST original")
x = np.float32(mnist.data)
x /= np.max(x, axis=1, keepdims=True)
y = np.int32(mnist.target)
mnist = chainer.datasets.TupleDataset(x, y)
mnist_train, mnist_test = chainer.datasets.split_dataset_random(mnist, 60000)

model = Classifier(MLP())

optimizer = chainer.optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(model)

train_iter = iterators.SerialIterator(mnist_train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(mnist_test, batch_size=100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (100, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()
