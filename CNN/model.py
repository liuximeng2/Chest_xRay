import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, pixel, in_channels=3, hidden_channels=(16, 32, 64), fc_sizes=(128, 1), 
                 pool_stride=2, reduction_factor=8, kernel_size=3, stride=1, padding=1):
        super(GraphConvolution, self).__init__()
        self.pixel = pixel
        self.reduction_factor = reduction_factor
        self.convoluted_pixel = pixel // self.reduction_factor
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0], 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2], 
                              kernel_size=kernel_size, stride=stride, padding=padding)

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=pool_stride)

        # Calculate conv_dim based on pooling operations
        out = self.pixel
        for _ in range(3):
            out = (out - kernel_size) // pool_stride + 1
        self.conv_dim = hidden_channels[2] * out * out

        self.fc1 = nn.Linear(self.conv_dim, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.conv_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
