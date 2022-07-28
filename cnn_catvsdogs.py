import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import MNIST
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Compose, Normalize
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import timm
from CatDogsDataset import CatDogsDataset


class CNNClassifier(pl.LightningModule):

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, padding_mode='reflect', stride=2),  #112x112
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  #28x28
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, 3, padding=1, stride=2), #7x7
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 3x3
            nn.Conv2d(256, 256, 3),  # 64x1x1
        )

        self.fc_model = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
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
        y_pred = self.forward(X)
        lossvalue = self.lossfunction(y_pred, y)
        y_pred_label = y_pred.argmax(dim=-1)
        accuracy = (y == y_pred_label).float().mean()
        if self.val_iteration % 500 == 0:
            sample_pred = y_pred[0].softmax(dim=-1)
            sample_pred = sample_pred.unsqueeze(0).unsqueeze(0)
            self.logger.experiment.add_image('sample_image', X[0], self.val_iteration)
            self.logger.experiment.add_image('vector_pred', sample_pred, self.val_iteration)
        self.val_iteration += 1
        return dict(loss=lossvalue.item(), accuracy=accuracy.item())

    def validation_epoch_end(self, outputs):
        val_avg_accuracy = [o['accuracy'] for o in outputs]
        val_avg_accuracy = sum(val_avg_accuracy) / len(val_avg_accuracy)
        val_avg_loss = np.mean([o['loss'] for o in outputs])
        self.log('val_loss', val_avg_loss, on_epoch=True)
        self.log('val/avg_accuracy', val_avg_accuracy, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class CNNClassifierPretrained(CNNClassifier):

    def __init__(self):
        super(CNNClassifierPretrained, self).__init__()
        self.conv_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='').eval()

        self.fc_model = nn.Sequential(
            nn.Linear(62720, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

        self.lossfunction = nn.CrossEntropyLoss()

        self.val_iteration = 0

    def forward(self, x):
        with torch.no_grad():
            conv_output = self.conv_model.forward_features(x)
            conv_output = torch.mean(conv_output, dim=[-2, -1])
        y_pred = self.fc_model(conv_output)
        return y_pred


def main():
    train_dataset = CatDogsDataset(train=True)
    test_dataset = CatDogsDataset(train=False)
    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=16, shuffle=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename='cnn-{epoch:02d}-{val_loss:.2f}')
    epochs = 10
    logger = TensorBoardLogger('cat_vs_dogs_logs')
    trainer = pl.Trainer(logger=logger, max_epochs=epochs, callbacks=[checkpoint_callback], gpus=1)
    model = CNNClassifierPretrained()
    trainer.fit(model, train_dl, test_dl)


if __name__ == '__main__':
    main()