import torch
import torch.nn as nn

class RuleALSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=4, num_layers=1):
        super(RuleALSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [バッチ(1), 時系列(32), 入力(5)]
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :]
        out = self.fc(last_step_out)
        return self.softmax(out)
