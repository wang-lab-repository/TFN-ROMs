import torch.nn as nn

mean_squared_error = nn.MSELoss()


class MyLoss(nn.Module):
    def __init__(self, partition):
        super(MyLoss, self).__init__()
        self.p = partition

    def forward(self, y_true, y_pred):
        # Customizing the computational logic of the loss function
        scalar = self.p * mean_squared_error(y_true[:, 0], y_pred[:, 0]) + \
                 (10 - self.p) * mean_squared_error(y_true[:, 1], y_pred[:, 1])
        return scalar
