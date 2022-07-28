import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles, make_s_curve
from sklearn.decomposition import PCA
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class AutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        decoded = self.decoder(latent_space)
        return dict(latent_space=latent_space, decoded=decoded)

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        pred_output = self(batch)
        x_reconstructed = pred_output['decoded']
        loss_value = ((batch - x_reconstructed) ** 2).sum(dim=1).mean(dim=0)
        return loss_value

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)


def main():
    batch_size = 32
    epochs = 500
    ae = AutoEncoder()
    X, _ = make_circles(2_000)



    X = torch.from_numpy(X).float()
    X = torch.cat([X[:1000,:], X[1000:, :1]], dim=1)
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0], X[:,1],X[:,2], cmap='Greens')
    plt.show()
    plt.close()

    dl = DataLoader(TensorDataset(X), batch_size)
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(ae, dl)

    with torch.no_grad():
        x_reduced_ae = ae.encoder(X)

    pca = PCA(n_components=2)
    x_reduced_pca = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)



    plt.plot(x_reduced_pca[:,0], x_reduced_pca[:,1], 'o')
    plt.title('PCA')
    plt.show()

    plt.plot(x_reduced_ae[:,0], x_reduced_ae[:,1], 'o')
    plt.title('AE')
    plt.show()

if __name__ == '__main__':
    main()