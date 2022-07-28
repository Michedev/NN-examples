import torch.nn
from torch.utils.data import Dataset
from path import Path
from torchvision.io import read_image
from torchvision.transforms import Resize, Pad, Normalize,  RandomApply, RandomAffine, ColorJitter, AutoAugmentPolicy
from torch.nn import AdaptiveAvgPool2d

class CatDogsDataset(Dataset):

    def __init__(self, train=True):
        data_path = Path(__file__).parent / 'data' / 'dogs_vs_cats'
        if train:
            data_path = data_path / 'train'
        else:
            data_path = data_path / 'test'
        self.path_images = data_path.files('*.jpg')
        self.resizer = AdaptiveAvgPool2d((224, 224))
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_aug = RandomApply(torch.nn.ModuleList([
            RandomAffine(10, (0.2, 0.2)),
            ColorJitter(0.3, 0.3, 0.3)
        ]))


    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, i):
        path_img = self.path_images[i]
        y = 0 if path_img.basename().startswith('cat') else 1
        img = read_image(path_img).float() / 255.0
        img = self.resizer(img)
        img = self.normalizer(img)
        img = self.data_aug(img)
        return img, y
