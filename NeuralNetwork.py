
from PIL import GimpGradientFile
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channel , 5x5 square convulation kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #the affine operator : y=wx +b
        self.fc1 = nn.Linear(16*5*5, 120)#5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        #convolution layer C1: 1 input image channel, 6 out
        #5x5 square convolution it uses RELU activation
        #outputs a tensor with size (N, 6, 28, 28)
        c1 = F.relu(self.conv1(input))
        #subsampling layer s2: 2x2 grid purely fuctional 
        #this layer does not have any parameters, and out puts a tensor with size (N, 6, 14, 14)
        s2 = F.max_pool2d(c1, (2,2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        #subsampling layer 4: 2x2 grid purely functional
        # this layer does not have any parameters, and outputs a tensor with 
        # size (N, 16, 5, 5)
        s4 = F.max_pool2d(c3, 2)
        #fatten operation: purely functional outputs (N, 400) tensors 
        s4 = torch.flatten(s4, 1)
        #fully connected layer F5: (N,400) tensor input 
        #outputs a (N, 120) tensor , it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Fully connected layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
         
        output = self.fc3(f6)
        return output

net = Net()
#print(net)

params = list(net.parameters())
#print(len(params))
#print(params[0].size()) # conv1's weight


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(f"output:{out}")

net.zero_grad()
out.backward(torch.randn(1, 10))

#eg of loss functions
output= net(input)
target= torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(f"loss:{loss}")

net.zero_grad()
loss.backward()
print(f"conv1 bias.grad:{net.conv1.bias.grad}")

#updating weights manually
#learning_rate =0.01
#for f in  net.parameters():
 #   with torch.no_grad():
 #       f -= f.grad * learning_rate

#to use various different update rules call torch.optim
#create optimizer
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
#in your trainning loop
optimizer.zero_grad()# zero the gradient buffers/clears the old gradient
output = net(input)#forward pass
loss = criterion(output, target)
loss.backward()#backward pass
optimizer.step()# does the update 


