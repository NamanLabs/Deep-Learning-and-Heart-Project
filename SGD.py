#SGD
import numpy as np
import matplotlib.pyplot as plt

f=lambda x,y:5*x**2+7*y**2
dx=lambda x:10*x
dy=lambda y:14*y

x=np.linspace(-5,5,100)
y=np.linspace(-5,5,100)
X,Y=np.meshgrid(x,y)
Z=f(X,Y)

x1,x2=-4.0,3.0
lr=0.05

x1_path,x2_path=[x1],[x2]

for _ in range(50):
    x1-=lr*dx(x1)
    x2-=lr*dy(x2)
    x1_path.append(x1)
    x2_path.append(x2)

print("The optimal value of x1 is:",x1)
print("The optimal value of x2 is:",x2)
print("The optimal value of y is:",f(x1,x2))

plt.contour(X,Y,Z,20)
plt.plot(x1_path,x2_path,'o-',color='red')
plt.title("SGD Optimization Path")
plt.show()