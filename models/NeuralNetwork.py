from torch.autograd import Variable as var
from extra.Utilities import *


class Model(nn.Module):
    def __init__(self, inp, output, hidden1, hidden2, hidden3, hidden4, hidden5):
        super().__init__()
        self.hidden1 = nn.Linear(inp, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.hidden3 = nn.Linear(hidden2, hidden3)
        self.hidden4 = nn.Linear(hidden3, hidden4)
        self.hidden5 = nn.Linear(hidden4, hidden5)
        self.output = nn.Linear(hidden5, output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))

        return self.output(x)


def calc_accuracy(tar_list, pred_list, out_len):
    inv_target = scaler.inverse_transform(np.array(tar_list))[:, -out_len:]
    inv_predict = scaler.inverse_transform(np.array(pred_list))[:, -out_len:]

    diff = abs(inv_predict - inv_target)
    divided = np.divide(diff, inv_target, out=np.zeros_like(diff), where=inv_target != 0)

    return 1 - np.array(divided)


def run_nn(data_set, epoch_num, hidden1, hidden2, hidden3, hidden4, hidden5):
    train_acc, valid_acc = [], []
    pred_len = 15
    
    train_x, train_y, val_x, val_y = gen_data(data_set)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        
    tmp = torch.zeros(len(scaler.data_max_) - pred_len).to(device)

    model = Model(len(train_x[0]), len(train_y[0]), hidden1, hidden2, hidden3, hidden4, hidden5).to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)

    for epoch in range(1, epoch_num):
        tar_list, pred_list = [], []
        model.train()

        print("Epoch: ", epoch)

        # Training the Model
        for x, y in zip(train_x, train_y):
            input_data = var(torch.from_numpy(x)).float().to(device)
            target = var(torch.from_numpy(y)).float().to(device)

            optimizer.zero_grad()
            predict = model(input_data)
            loss = loss_function(predict, target)
            loss.backward()
            optimizer.step()

            target = torch.cat([tmp, target.detach()], dim=0)
            predict = torch.cat([tmp, predict.detach()], dim=0)

            tar_list.append(target.cpu().numpy())
            pred_list.append(predict.cpu().numpy())

        train_acc.append(calc_accuracy(tar_list, pred_list, pred_len).mean(0))

        tar_list, pred_list = [], []
        model.eval()

        # Validating the Model
        for x, y in zip(val_x, val_y):
            input_data = var(torch.from_numpy(x)).float().to(device)
            target = var(torch.from_numpy(y)).float().to(device)

            predict = model(input_data)
            
            target = torch.cat([tmp, target.detach()], dim=0)
            predict = torch.cat([tmp, predict.detach()], dim=0)

            tar_list.append(target.cpu().numpy())
            pred_list.append(predict.cpu().numpy())

        valid_acc.append(calc_accuracy(tar_list, pred_list, pred_len).mean(0))

        print("Train Acc:", train_acc[-1])
        print("Val Acc:", valid_acc[-1])

    return train_acc, valid_acc


if __name__ == "__main__":

    data = sliding_window(10, out_num=1)
    train, val = run_nn(data, 150, 200, 160, 110, 70, 30)
    pd.DataFrame(train).to_csv('../results/nn/tr_cross_val.csv', index=False)
    pd.DataFrame(val).to_csv('../results/nn/val_cross_val.csv', index=False)
