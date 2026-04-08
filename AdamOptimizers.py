y = 5.0
lr = 0.1
epochs = 5
w = 10.0
m = 0.0
v = 0.0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

for epoch in range(epochs):
    grad = 2 * (w - y)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    w = w - lr * m / ((v ** 0.5) + eps)
    print("Epoch", epoch+1, "Weight:", w)
