import torch.nn as nn

# convolutional neural network
class CNN(nn.Module):
    def __init__(self, classes):
        super(CNN, self).__init__()
        # 28 14 ->  32  16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 14 7  -> 16   8
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 8 4
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # 4 2
        self.layer4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc1 = nn.Linear(1*1*512,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out