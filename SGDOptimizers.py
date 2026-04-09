y = 5.0
lr = 0.1
epochs = 5
w = 10.0

for epoch in range(epochs):
    grad = 2 * (w - y)
    w = w - lr * grad
    print("Epoch", epoch+1, "Weight:", w)
