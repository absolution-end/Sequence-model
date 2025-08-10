# MNIST-Image data
# DataLoader and Transformation
# Multilayer Neural Net and Activation function
# Loss & Optimizer
# Training Loop (Batch training)
# Model evaluation
# GPU support


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameter
# input_size = 800
hidden_size = 32
num_epoch = 10
num_classes = 10
batch_size = 4
learning_rate = 0.001

# Dataset has PTLImages from range [0,1]
# we Transform them to tensor of normilazation range [-1,1]
transforms = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((.5,.5,.5),(.5,.5,.5))]) # (R,G,B)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transforms)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog','horse','ship','truck')
example = iter(train_loader)
samples, label =next(example)

# print(samples.shape, label.shape)
for i in range(len(samples)):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i][0])
plt.show()

# implement conv net

class CNN_RNN_GRU_LSTM(nn.Module):
        def __init__(self, hidden_size, num_classes):
                super(CNN_RNN_GRU_LSTM,self).__init__()
                self.cnn = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),  # 32x16x16
                            nn.Conv2d(32, 64, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2)   # 64x8x8
                        )
                #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                
                # RNN
                self.hidden_size = hidden_size
                self.rnn = nn.RNN(input_size=64*8,hidden_size=hidden_size,num_layers=2,batch_first=True)
                self.fc = nn.Linear(hidden_size,num_classes)
                
        def forward(self, x):
                batch_size = x.size(0)
        
                # CNN
                features = self.cnn(x)  # [batch, 64, 8, 8]
                
                # Prepare sequence: treat height=8 as time steps
                features = features.permute(0, 2, 1, 3)  # [batch, height=8, channels=64, width=8]
                features = features.reshape(batch_size, 8, -1)  # [batch, seq_len=8, input_size=64*8]
                
                # RNN
                out, _ = self.rnn(features)  # [batch, seq_len, hidden_size]
                out = out[:, -1, :]  # last time step
                
                # Classifier
                out = self.fc(out)
                return out
        
model = CNN_RNN_GRU_LSTM(hidden_size=hidden_size, num_classes=num_classes).to(device)

# loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epoch):
        for i , (image, label) in enumerate(train_loader):
                
                # origin shape: [4,3,32,32] = 4,3,1024
                # input layer: 3 input channel, 6 output channel, 5 kernal channel
                image = image.to(device)
                label = label.to(device)
                
                # forward pass
                output = model(image)
                loss = criterion(output, label)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (1+i) %2000 ==0:
                        print(f'epoch [{epoch+1}/{num_epoch}], step[{i+1}/{n_total_steps}], [loss: {loss.item():.4f}]')
                        
print("finish training")

with torch.no_grad():
        n_correct =0
        n_samples =0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for image, labels in test_loader:
                image = image.to(device)
                label = labels.to(device)
                output = model(image)
                
                # max returns (value, index)
                _, predictions = torch.max(output,1)
                n_correct = (predictions==labels).sum().item()
                n_samples = labels.size(0)
                
                for i in range(batch_size):
                        label= labels[i]
                        pred = predictions[i]
                        if (label == pred):
                                n_class_correct[label] +=1
                                n_class_samples[label] +=1
                                
        acc = 100.0 * n_correct /n_samples
        print(f'Accuracy of the network:{acc}%')
        
        for i in range(10):
                acc = 100.0 * n_class_correct[i] /n_class_samples[i]
                print(f"Accuracy of {classes[i]}:{acc}%")