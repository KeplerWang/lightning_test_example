from lightning import LightningModule, LightningDataModule, Trainer
from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset


class IndexDataset(Dataset):
    def __init__(self, length: int = 2048):
        super().__init__()
        self.data = list(range(length))

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.data[idx], dtype=torch.float)

    def __len__(self):
        return len(self.data)


class IndexDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = IndexDataset(32)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=4,
            num_workers=4,
        )


class IndexModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)

    def forward(self, x):
        return self.model(x.unsqueeze(1))

    def training_step(self, batch, batch_idx):
        idx, x = batch
        y = self(x)
        loss = nn.functional.mse_loss(y.squeeze(), x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
