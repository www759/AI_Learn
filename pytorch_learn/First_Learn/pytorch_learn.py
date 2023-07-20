import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import iris_dataloader

class nn_learn(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# Load and Split Data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

custom_dataset = iris_dataloader("./pytorch_learn/data.txt")
train_size = int(len(custom_dataset) * 0.7)
val_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("train_size: ", train_size, " val_size: ", val_size, " test_size", test_size)


# compute and return accuracy
def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()
    acc = acc_num / len(dataset)
    return acc

def main(lr=0.005, epochs = 20):
    model = nn_learn(4, 12, 6, 3)
    loss_f = nn.CrossEntropyLoss()

    # model's weights
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=lr)

    # weights' save path
    save_path = os.path.join(os.getcwd(), "results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1)
        sample_num = 0

        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for datas in train_bar:
            data, label = datas
            label = label.squeeze(-1)
            sample_num = data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1] # 返回值是一个元组，第一个元素是值，第二个是值的索引
            acc_num = torch.equal(pred_class, label.to(device))

            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        val_acc = infer(model, val_loader, device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f} val_acc{:.3f}".format(epoch+1, epochs, loss, train_acc, val_acc))
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # qing 0
        train_acc = 0
        val_acc = 0
    print("Finished Training")

    test_acc = infer(model, test_loader, device)
    print("test_acc: ", test_acc)

if __name__ == "__main__":
    main()