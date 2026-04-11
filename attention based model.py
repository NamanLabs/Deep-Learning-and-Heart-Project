import torch
import torch.nn as nn
import torch.optim as optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM=50
OUTPUT_DIM=50
EMB_DIM=32
HID_DIM=64
SEQ_LEN=8
BATCH_SIZE=32

class Encoder(nn.Module):
    def __init__(self,input_dim,emb_dim,hid_dim):
        super().__init__()
        self.embedding=nn.Embedding(input_dim,emb_dim)
        self.rnn=nn.LSTM(emb_dim,hid_dim,batch_first=True)

    def forward(self,x):
        embedded=self.embedding(x)
        outputs,(hidden,cell)=self.rnn(embedded)
        return outputs,hidden,cell


class Attention(nn.Module):
    def __init__(self,hid_dim):
        super().__init__()
        self.attn=nn.Linear(hid_dim*2,hid_dim)
        self.v=nn.Linear(hid_dim,1,bias=False)

    def forward(self,hidden,encoder_outputs):
        seq_len=encoder_outputs.shape[1]
        hidden=hidden[-1].unsqueeze(1).repeat(1,seq_len,1)
        energy=torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        attention=self.v(energy).squeeze(2)
        return torch.softmax(attention,dim=1)


class Decoder(nn.Module):
    def __init__(self,output_dim,emb_dim,hid_dim,attention):
        super().__init__()
        self.embedding=nn.Embedding(output_dim,emb_dim)
        self.rnn=nn.LSTM(hid_dim+emb_dim,hid_dim,batch_first=True)
        self.fc=nn.Linear(hid_dim,output_dim)
        self.attention=attention

    def forward(self,input,hidden,cell,encoder_outputs):
        input=input.unsqueeze(1)
        embedded=self.embedding(input)

        attn_weights=self.attention(hidden,encoder_outputs)
        attn_weights=attn_weights.unsqueeze(1)

        context=torch.bmm(attn_weights,encoder_outputs)

        rnn_input=torch.cat((embedded,context),dim=2)

        output,(hidden,cell)=self.rnn(rnn_input,(hidden,cell))

        prediction=self.fc(output.squeeze(1))

        return prediction,hidden,cell


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,src,trg):
        encoder_outputs,hidden,cell=self.encoder(src)

        outputs=torch.zeros(trg.shape[0],trg.shape[1],OUTPUT_DIM).to(device)

        input=trg[:,0]

        for t in range(1,trg.shape[1]):
            output,hidden,cell=self.decoder(input,hidden,cell,encoder_outputs)
            outputs[:,t]=output
            input=output.argmax(1)

        return outputs


encoder=Encoder(INPUT_DIM,EMB_DIM,HID_DIM)
attention=Attention(HID_DIM)
decoder=Decoder(OUTPUT_DIM,EMB_DIM,HID_DIM,attention)

model=Seq2Seq(encoder,decoder).to(device)

optimizer=optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()

for epoch in range(5):

    src=torch.randint(0,INPUT_DIM,(BATCH_SIZE,SEQ_LEN)).to(device)
    trg=torch.randint(0,OUTPUT_DIM,(BATCH_SIZE,SEQ_LEN)).to(device)

    optimizer.zero_grad()

    output=model(src,trg)

    output_dim=output.shape[-1]
    output=output[:,1:].reshape(-1,output_dim)
    trg=trg[:,1:].reshape(-1)

    loss=criterion(output,trg)
    loss.backward()
    optimizer.step()

    print("Epoch:",epoch+1,"Loss:",loss.item())