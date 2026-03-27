import numpy as np
import matplotlib.pyplot as plt

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

def sig(x): return 1/(1+np.exp(-x))
def dsig(x): return sig(x)*(1-sig(x))

W1=np.random.randn(2,3)
W2=np.random.randn(3,1)

lr=0.1
losses=[]

for i in range(2000):
    z1=X@W1
    h=sig(z1)
    z2=h@W2
    y_pred=sig(z2)

    loss=np.mean((y-y_pred)**2)
    losses.append(loss)

    e=y_pred-y
    dW2=h.T@(e*dsig(z2))
    dW1=X.T@((e*dsig(z2))@W2.T*dsig(z1))

    W2-=lr*dW2
    W1-=lr*dW1

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

print("Final Predictions:\n",np.round(y_pred,3))
