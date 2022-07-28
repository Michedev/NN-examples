import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import MNIST
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class CNNClassifier(pl.LightningModule):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #16x14x14
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x7x7
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x3x3
            nn.Conv2d(32, 64, 3), # 64x1x1
        )

        self.fc_model = nn.Sequential(
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

        self.lossfunction = nn.CrossEntropyLoss()

        self.val_iteration = 0

    def forward(self, x):
        conv_output = self.conv_model(x)
        conv_output = conv_output.flatten(1)
        y_pred = self.fc_model(conv_output)
        return y_pred

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.forward(X)
        lossvalue = self.lossfunction(y_pred, y)
        y_pred_label = y_pred.argmax(dim=-1)
        accuracy = (y == y_pred_label).float().mean()
        self.log('train_batch_acc', accuracy, prog_bar=True, logger=False)
        return dict(loss=lossvalue, accuracy=accuracy)
    
    def training_epoch_end(self, outputs):
        train_avg_accuracy = [o['accuracy'] for o in outputs]
        train_avg_accuracy = sum(train_avg_accuracy) / len(train_avg_accuracy)
        self.log('train/avg_accuracy', train_avg_accuracy, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        self.logger.experiment.add_image('sample_image', X[0], self.val_iteration)
        y_pred = self.forward(X)
        lossvalue = self.lossfunction(y_pred, y)
        y_pred_label = y_pred.argmax(dim=-1)
        accuracy = (y == y_pred_label).float().mean()
        self.val_iteration += 1
        sample_pred = y_pred[0].softmax(dim=-1)
        sample_pred = sample_pred.unsqueeze(0).unsqueeze(0)
        self.logger.experiment.add_image('vector_pred', sample_pred, self.val_iteration)
        return dict(loss=lossvalue.item(), accuracy=accuracy.item())
    
    def validation_epoch_end(self, outputs):

        val_avg_accuracy = [o['accuracy'] for o in outputs]
        val_avg_accuracy = sum(val_avg_accuracy) / len(val_avg_accuracy)
        val_avg_loss = np.mean([o['loss'] for o in outputs])
        self.log('val_loss', val_avg_loss, on_epoch=True)
        self.log('val/avg_accuracy', val_avg_accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():
    train_dataset = MNIST('data', download=True, transform=ToTensor())
    test_dataset = MNIST('data', train=False, download=True, transform=ToTensor())
    train_dl = DataLoader(train_dataset, batch_size=32)
    test_dl = DataLoader(test_dataset, batch_size=32)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename='cnn-{epoch:02d}-{val_loss:.2f}')
    epochs = 10
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], gpus=1)
    model = CNNClassifier()
    trainer.fit(model, train_dl, test_dl)

if __name__ == '__main__':
    main()