import PIL.Image
import torch
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt

from cnn_mnist import CNNClassifier

@torch.no_grad()
def main():
    adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((28,28))
    checkpoint_model = 'lightning_logs/version_2/checkpoints/cnn-epoch=06-val_loss=0.04.ckpt'
    model_trained = CNNClassifier.load_from_checkpoint(checkpoint_model)
    myimage = read_image('examples/8_big.png', ImageReadMode.GRAY)
    myimage = myimage.unsqueeze(0)
    myimage = myimage.float() / 255.0
    myimage = 1 - myimage
    if myimage.shape[-2:] != (28, 28):
        myimage = adaptive_avg_pool(myimage)
        plt.imshow(myimage[0,0], cmap='Greys')
        plt.show()
        plt.close()
    model_trained.eval()
    y_pred = model_trained.forward(myimage)
    print(y_pred)
    print(y_pred.softmax(dim=-1))
    plt.bar(list(range(10)), y_pred.softmax(dim=-1).squeeze(0).numpy())
    plt.show()
    print(y_pred.argmax(dim=-1))

if __name__ == '__main__':
    main()