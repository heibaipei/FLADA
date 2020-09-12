import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from torch.autograd import Variable
class DCD(BasicModule):
    def __init__(self,h_features=64, input_features=128):
        super(DCD,self).__init__()

        self.fc1=nn.Linear(input_features,h_features)
        self.fc2=nn.Linear(h_features,h_features)
        self.fc3=nn.Linear(h_features,4)

    def forward(self,inputs):
        out=F.relu(self.fc1(inputs))
        out=self.fc2(out)
        return F.softmax(self.fc3(out),dim=1)

class Classifier(BasicModule):
    def __init__(self,input_features=64):
        super(Classifier,self).__init__()
        self.fc=nn.Linear(input_features, 2)   ##  分类种类

    def forward(self, input):
        return F.softmax(self.fc(input),dim=1)

class Encoder(BasicModule):
    def __init__(self, batch_size, output_size, hidden_size, embed_dim, bidirectional, dropout, use_cuda, attention_size, sequence_length):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim1 = embed_dim  ### space is 64,temple is
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length1 = sequence_length
        # self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        # self.lookup_table.weight.data.uniform_(-1., 1.)
        self.layer_size = 1
        self.hidden_size2 = hidden_size
        self.embed_dim2 = sequence_length
        self.sequence_length2 = embed_dim
        self.dt = nn.Dropout(self.dropout)
        self.lstm2 = nn.LSTM(self.embed_dim2,
                             self.hidden_size2,
                             self.layer_size,
                             dropout=self.dropout,
                             bidirectional=self.bidirectional)

        self.lstm1 = nn.LSTM(self.embed_dim1,
                             self.hidden_size,
                             self.layer_size,
                             dropout=self.dropout,
                             bidirectional=self.bidirectional)
        self.bn = nn.BatchNorm2d(1, affine=False, momentum=1)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.randn(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.randn(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.randn(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.randn(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size * 2, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length1])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length1, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def attention_net2(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length2])
        # print(exps.size()) = (batch_size, squence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length2, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, x, batch_size=None):
        input_ = x.float()  ## batch, 128, 32
        # print(input_.shape)
        input_ = self.bn(input_)
        input_ = torch.squeeze(input_, 1)

        input1 = input_.permute(1, 0, 2)  # (seq_len, batch, input_size)  ## 128, batch, 32
        # print(input1.shape)
        input2 = input_.permute(2, 0, 1)

        if self.use_cuda:
            h_0 = Variable(torch.randn(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.randn(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.randn(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.randn(self.layer_size, self.batch_size, self.hidden_size))
        # print("h_0.shape", h_0.shape)
        # print("c_0.shape", c_0.shape)
        # print("self.batch_size", self.batch_size)
        lstm_output1, (final_hidden_state, final_cell_state) = self.lstm1(input1, (h_0, c_0))
        attn_output1 = self.attention_net(lstm_output1)
        lstm_output2, (final_hidden_state, final_cell_state) = self.lstm2(input2, (h_0, c_0))
        attn_output2 = self.attention_net2(lstm_output2)
        # print(attn_output1.shape)
        # print(attn_output2.shape)
        all_features = torch.cat((attn_output1, attn_output2), dim=1)  ## batch_size, hidden_size*layer_size
        # all_features = attn_output1 + attn_output2
        # print(all_features.shape)  ### 59*128
        # all_features = all_features.reshape(batch_size, -1)
        logits = self.label(all_features)
        logits = self.dt(logits)
        return logits




    #     self.conv1=nn.Conv2d(1, 6, 5)
    #     self.conv2=nn.Conv2d(6, 16, 5)
    #     self.fc1=nn.Linear(2320 , 1000)
    #     self.fc2=nn.Linear(1000 , 84)
    #     self.fc3=nn.Linear(84 , 64)
    #
    # def forward(self,input):
    #     input = input.float()
    #     out=F.relu(self.conv1(input))
    #     out=F.max_pool2d(out,2)
    #     out=F.relu(self.conv2(out))
    #     out=F.max_pool2d(out,2)
    #     out=out.view(out.size(0),-1)
    #
    #     out=F.relu(self.fc1(out))
    #     out=F.relu(self.fc2(out))
    #     out=self.fc3(out)
    #
    #     return out





