import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import config


class Net(nn.Module):
    def __init__(
        self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


torch.manual_seed(config["train_eval"].get("seed"))
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config["train_eval"].get("learning_rate")
)
