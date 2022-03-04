import argparse
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from cnn_model import CNN
from torch_dataset import dataset
import matplotlib.pyplot as plt

# use GPA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
def get_data(datapath, batch_size):
    # get the original data
    data = pd.read_csv(datapath)
    # print(data)
    data = data.sample(frac=1)
    # 80% data for training, 20% data for testing
    train_index = int(0.8 * len(data))
    train_dataset = data[0:train_index]
    test_dataset = data[train_index:len(data)]
    # create the dataset that suits torch (I define it in torch_dataset.py)
    train_dataset = dataset(train_dataset)
    test_dataset = dataset(test_dataset)
    print(type(train_dataset))
    # load data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader



# train
def train(epochs, learning_rate, train_loader):
    # train_loader = train_loader.view(-1,1,28,28)
    criter = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    total_step = len(train_loader)
    # print(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).view(-1,1,28,28)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criter(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch+1,epochs,i+1,total_step,loss.item()))
            i += 1
    return model

# test
def test(model,test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device).view(-1,1,28,28)
            labels = labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
        # print('Test Accuracy of the model on the test data: {} %'.format(100 * correct / total))
    return correct / total


# produce
def predict(model,vali_loader):
    model.eval()
    res = []

    with torch.no_grad():
        for images, labels in vali_loader:
            images = images.to(device).view(-1,1,28,28)
            labels = labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data,1)
            res.extend(predict.view(-1).cpu().numpy().tolist())
    # print(res[0].values)
    # print(predict)
    pd.DataFrame(res).to_csv('cnn_pred.csv',index=False,encoding="utf-8")
# print(model)
# save model
# torch.save(model.state_dict(),'cnn_test_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the CNN:')
    parser.add_argument('epochs', metavar='epochs', type=int, help='iteration',default=10)
    parser.add_argument('batch_size', metavar='batch_size', type=int,default=256,help='number of samples')
    parser.add_argument('learning_rate', metavar='learning_rate', type=float, default=0.001,help='learning rate')
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    classes = 10
    datapath = 'training.csv/training.csv'
    predict_path = 'testing.csv/testing.csv'
    data = pd.read_csv(datapath)
    # define the model
    model = CNN(classes=classes).to(device)
    # 5-folds cross validation
    kf = KFold(n_splits=5)
    t = 1
    acc = []
    print("Using 5-folds cross validation:")
    for train_index, test_index in kf.split(data):
        train_dataset, test_dataset = data.iloc[train_index, :], data.iloc[test_index, :]
        train_dataset = dataset(train_dataset)
        test_dataset = dataset(test_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
        model = train(epochs, learning_rate, train_loader)
        temp = test(model, test_loader)
        print("Folds: ",t)
        print("Acc: ",temp)
        print('-----------------------------------')
        t += 1
        acc.append(temp)

    print("Mean acc: {0}".format(np.mean(acc)))

    vali_data = pd.read_csv(predict_path)
    new_column = np.zeros((len(vali_data),1))
    vali_data.insert(1,'label',new_column)
    vali_data = dataset(vali_data)
    vali_loader = torch.utils.data.DataLoader(dataset=vali_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    predict(model,vali_loader)

    plt.plot(range(1,6), acc)
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.ylim((0.8, 1.0))
    plt.grid()
    plt.show()


