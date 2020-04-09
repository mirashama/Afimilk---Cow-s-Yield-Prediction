from extra.Utilities import *


class LSTM(nn.Module):

    def __init__(self, device, input_size, hidden_dim, out_size, num_layers=1, drop_out=0.2):
        super(LSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lin = nn.Linear(hidden_dim, out_size)

        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = nn.Dropout(drop_out)
        self.func = nn.Tanh()

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm1(x, hidden)
        out = self.dropout(lstm_out)
        lstm_out = self.lin(out)
       
        return self.func(lstm_out), hidden

    def init_hidden(self, sw_num):
        return ((torch.zeros((self.n_layers, sw_num, self.hidden_dim))).to(self.device),
                (torch.zeros((self.n_layers, sw_num, self.hidden_dim))).to(self.device))


def calc_accuracy(pred_list, tar_list):
    pred_list, tar_list = np.array(pred_list), np.array(tar_list)
    diff = abs(pred_list - tar_list)
    val = np.divide(diff, tar_list, out=np.zeros_like(diff), where=tar_list != 0)

    return 1 - np.array(val)


# split to train and test - 90 - 10 %
def split_data(data):
    global_train = []
    global_test = []

    for cow in data:
        cross = len(cow)
        tmp = cow[:math.floor(0.9 * cross)]
        tmp2 = cow[math.floor(0.9 * cross):]

        for val in tmp:
            global_train.append(val)
        for val in tmp2:
            global_test.append(val)

    return global_train, global_test


def lstm(input_data, output_data, sw_num, lr, hid_layer):
    torch.manual_seed(1)

    train_x, test_x = split_data(input_data)
    train_y, test_y = split_data(output_data)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = LSTM(device,
                 input_size=len(input_data[0][0][0]),
                 hidden_dim=64,
                 out_size=len(output_data[0][0][0]),
                 num_layers=hid_layer).to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    acc_train, acc_test = [], []
    
    for epoch in range(100):
        model.train()
        print("Epoch: ", epoch)

        j = 0
        old_cow = -1
        for i in range(len(train_x)):
            new_cow = train_x[i][-1][2]

            if old_cow != new_cow:
                old_cow = new_cow
                hidden = model.init_hidden(sw_num)

            model.zero_grad()

            prediction, hidden = model(torch.FloatTensor([train_x[i]]).to(device), tuple(h.detach() for h in hidden))

            loss = loss_function(prediction, torch.FloatTensor([train_y[i]]).to(device))
            loss.backward()
            optimizer.step()

            prediction = prediction.cpu()
            prediction = prediction.tolist()

            if j == 0:
                acc = calc_accuracy(prediction[0][-1], train_y[i][-1])
                j = 1
            else:
                acc = np.vstack([acc, calc_accuracy(prediction[0][-1], train_y[i][-1])])

        acc_train.append(acc.mean(0))
        print("Train Acc:", np.round(acc.mean(0), 3))

        model.eval()
        j = 0
        for i in range(len(test_x)):
            new_cow = test_x[i][-1][2]
            if old_cow != new_cow:
                old_cow = new_cow                
                hidden = model.init_hidden(sw_num)
           
            prediction, hid = model(torch.FloatTensor([test_x[i]]).to(device), tuple(h.detach() for h in hidden))
            prediction = prediction.cpu()
            prediction = prediction.tolist()

            if j == 0:
                acc = calc_accuracy(prediction[0][-1], test_y[i][-1])
                j = 1
            else:
                acc = np.vstack([acc, calc_accuracy(prediction[0][-1], test_y[i][-1])])
        
        acc_test.append(acc.mean(0))
        print("Val Acc:", np.round(acc.mean(0), 3))
    
    return acc_train, acc_test


if __name__ == "__main__":
    # output will have 5 features
    inp, out = sliding_window_lstm(1, mean=1)
    train, test = lstm(inp, out, 1, 0.0001, 3)
    pd.DataFrame(train).to_csv('../results/lstm/tr_lstm_mean.csv', index=False)
    pd.DataFrame(test).to_csv('../results/lstm/val_lstm_mean.csv', index=False)

