from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(1, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Conv2d(64, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Conv2d(128, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(.2)

        )

        self.pred = nn.Sequential(

            nn.Linear(1152, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Linear(512, 7),
            nn.Softmax()

        )

    def forward(self, img):
        pred = self.main(img)
        pred = self.pred(pred.view(-1, 1152))
        return pred
