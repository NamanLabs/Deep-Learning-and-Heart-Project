import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.ToTensor()
dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transform,download=True)
loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True)

class SelfAttention(nn.Module):
    def __init__(self,embed_size):
        super(SelfAttention,self).__init__()
        self.query=nn.Linear(embed_size,embed_size)
        self.key=nn.Linear(embed_size,embed_size)
        self.value=nn.Linear(embed_size,embed_size)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        attention=self.softmax(torch.bmm(Q,K.transpose(1,2)))
        out=torch.bmm(attention,V)
        return out

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel,self).__init__()
        self.attention=SelfAttention(28)
        self.fc=nn.Linear(28*28,10)

    def forward(self,x):
        x=x.squeeze(1)
        x=self.attention(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc(x)
        return x

model=AttentionModel().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=5

for epoch in range(epochs):
    for images,labels in loader:

        images=images.to(device)
        labels=labels.to(device)

        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:",epoch+1,"Loss:",loss.item())