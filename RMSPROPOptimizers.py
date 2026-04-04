y = 5.0
lr = 0.1
epochs = 5
w = 10.0
v = 0.0
beta = 0.9
eps = 1e-8

for epoch in range(epochs):
    grad = 2 * (w - y)
    v = beta * v + (1 - beta) * (grad ** 2)
    w = w - lr * grad / ((v ** 0.5) + eps)
    print("Epoch", epoch+1, "Weight:", w)
