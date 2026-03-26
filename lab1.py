def train_perceptron():
    inputs=[(0,0),(0,1),(1,0),(1,1)]
    targets=[0,0,0,1]
    w1=0
    w2=0
    b=0
    lr=0.1
    for epoch in range(20):
        for i in range(len(inputs)):
            x1,x2=inputs[i]
            z=w1*x1+w2*x2+b
            output=1 if z>=0 else 0
            error=targets[i]-output
            w1+=lr*error*x1
            w2+=lr*error*x2
            b+=lr*error
    return w1,w2,b

w1,w2,b=train_perceptron()

print("x1 x2 | AND Output")
print("-------------------")
for x1,x2 in [(0,0),(0,1),(1,0),(1,1)]:
    z=w1*x1+w2*x2+b
    output=1 if z>=0 else 0
    print(x1,x2," | ",output)
