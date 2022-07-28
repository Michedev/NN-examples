import numpy as np
import torch
from sklearn.datasets import make_moons
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

def make_nn():
    model = nn.Sequential(
        nn.Linear(2, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
        nn.Sigmoid()
    )
    return model

def main():
    model = make_nn()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossfunction = torch.nn.BCELoss()
    X_train, y_train = make_moons(n_samples=1_000, random_state=13, noise=0.01)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).unsqueeze(1).float()
    train_dataset = TensorDataset(X_train, y_train)
    
    X_test, y_test = make_moons(n_samples=1_000, random_state=28, noise=0.01)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).unsqueeze(1).float()
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    epochs = 100
    dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    accuracy_test = []
    for epoch in range(epochs):
        print('epoch', epoch + 1)
        epoch_loss = []
        for X_batch, y_batch in iter(dl):
            y_pred = model(X_batch)
            loss_value = lossfunction(y_pred, y_batch)
            loss_value.backward()
            opt.step()
            opt.zero_grad()
            epoch_loss.append(loss_value.item())
        print('avg loss: ', sum(epoch_loss) / len(epoch_loss))

        epoch_loss_test = []
        epoch_accuracy_test = []
        model.eval()
        for X_test_batch, y_test_batch in test_dl:
            with torch.no_grad():
                y_pred = model(X_test_batch)
            loss_value = lossfunction(y_pred, y_test_batch).item()
            epoch_loss_test.append(loss_value)
            y_label_test = (y_pred > 0.5).long()
            accuracy = (y_test_batch == y_label_test).float()
            accuracy = sum(accuracy) / len(accuracy)
            epoch_accuracy_test.append(accuracy.item())
        print('avg test loss:', sum(epoch_loss_test) / len(epoch_loss_test))
        avg_epoch_accuracy_test = sum(epoch_accuracy_test) / len(epoch_accuracy_test)
        print('avg test accuracy:', avg_epoch_accuracy_test)
        accuracy_test.append(avg_epoch_accuracy_test)
        model.train()

    plt.plot(accuracy_test, '-o')
    plt.title("Test accuracy")
    plt.show()

    x1 = torch.linspace(-3.0, 3.0, 100)
    x2 = torch.linspace(-3.0, 3.0, 100)

    grid_x1, grid_x2 = torch.meshgrid(x1, x2)

    X = torch.stack((grid_x1, grid_x2), dim=-1)
    X = X.flatten(0, 1)

    with torch.no_grad():
        y_pred = model(X)
        y_pred = y_pred.reshape(grid_x1.shape)
    plt.contourf(grid_x1, grid_x2, y_pred)
    sns.scatterplot(X_test[:, 0], X_test[:, 1], hue=y_test.squeeze(1))
    plt.show()
    

if __name__ == '__main__':
    main()

