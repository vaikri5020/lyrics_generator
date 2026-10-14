#  model building 
# X ka size 82554, 1539
# y ka size 82554
# batch ki(128,82554) 128 batch size aur har ek batch ma 82554
#  har word ko vector 
#  embedding layer ka pass karenga (128,82554,100)-> yeh tensor into lstm
#  100 dimensions ka vector embedding

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

vocab=8800

class LSTMModel(nn.Module):
    def __init__(self,vocab_size,embed_dim=100):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim)
        self.lstm=nn.LSTM(100,150,batch_first=True)
        self.fc=nn.Linear(150,vocab_size)

    def forward(self,x):
        embedded=self.embedding(x)
        intermediate_state,(final_hidden_state,final_cell_state)=self.lstm(embedded)
        output=self.fc(final_hidden_state.squeeze(0))
        return output


model=LSTMModel(8800)    

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

epochs=50
lr_rate=0.001
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate)



