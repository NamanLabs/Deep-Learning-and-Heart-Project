#Adam 
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
lr,b1,b2,eps=0.1,0.9,0.999,1e-8
m1=m2=v1=v2=0

x1_path,x2_path=[x1],[x2]

for t in range(1,51):
    g1,g2=dx(x1),dy(x2)
    m1=b1*m1+(1-b1)*g1
    v1=b2*v1+(1-b2)*(g1**2)
    m2=b1*m2+(1-b1)*g2
    v2=b2*v2+(1-b2)*(g2**2)

    m1h=m1/(1-b1**t)
    v1h=v1/(1-b2**t)
    m2h=m2/(1-b1**t)
    v2h=v2/(1-b2**t)

    x1-=lr*m1h/np.sqrt(v1h+eps)
    x2-=lr*m2h/np.sqrt(v2h+eps)
    x1_path.append(x1)
    x2_path.append(x2)

print("The optimal value of x1 is:",x1)
print("The optimal value of x2 is:",x2)
print("The optimal value of y is:",f(x1,x2))

plt.contour(X,Y,Z,20)
plt.plot(x1_path,x2_path,'o-',color='red')
plt.title("Adam Optimization Path")
plt.show()
