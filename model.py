# 搭建神经网络
import torch
from torch import nn


class Xiaozu(nn.Module):
    def __init__(self):
        super(Xiaozu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 256, 17, 1, 8),  # 256 @ 256 * 256
            nn.MaxPool2d(4),  # 256 @ 64 * 64
            nn.Conv2d(256, 256, 17, 1, 8),  # 256 @ 64 * 64
            nn.MaxPool2d(4),  # 256 @ 16 * 16
            nn.Flatten(),  # 256 * 16 * 16
            nn.Linear(256 * 16 * 16, 256),  # 256
            nn.Linear(256, 8),  # 8
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    print("hello")
    print(torch.cuda.is_available())
    xiaozu = Xiaozu()
    input = torch.ones((64, 3, 256, 256))
    output = xiaozu(input)
    print(output.shape)