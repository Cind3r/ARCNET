class TimeSeriesAdapter(nn.Module):
    def __init__(self, in_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x):
        return self.encoder(x)
