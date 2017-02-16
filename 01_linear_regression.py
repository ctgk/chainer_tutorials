import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# training data
x_train = np.float32(np.linspace(-1, 1, 11)[:, None])
y_train = np.float32(2 * x_train + np.random.randn(*x_train.shape) * 0.33)

# y = Xw + b
model = L.Linear(1, 1)

optimizer = chainer.optimizers.SGD(lr=0.01)
optimizer.use_cleargrads()  # for efficiency
optimizer.setup(model)

for i in range(1000):
    model.cleargrads()
    y_pred = model(x_train)
    loss = F.mean_squared_error(y_pred, y_train)
    loss.backward()
    optimizer.update()

print(model.W.data)  # This should be about 2.
print(model.b.data)  # This should be about 0.
