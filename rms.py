#RMSProp
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
lr,gamma,eps=0.1,0.9,1e-8
e1=e2=0

x1_path,x2_path=[x1],[x2]

for _ in range(50):
    g1,g2=dx(x1),dy(x2)
    e1=gamma*e1+(1-gamma)*g1**2
    e2=gamma*e2+(1-gamma)*g2**2
    x1-=lr*g1/np.sqrt(e1+eps)
    x2-=lr*g2/np.sqrt(e2+eps)
    x1_path.append(x1)
    x2_path.append(x2)

print("The optimal value of x1 is:",x1)
print("The optimal value of x2 is:",x2)
print("The optimal value of y is:",f(x1,x2))

plt.contour(X,Y,Z,20)
plt.plot(x1_path,x2_path,'o-',color='red')
plt.title("RMSprop Optimization Path")
plt.show()
