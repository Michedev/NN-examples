from path import Path
from tqdm import tqdm
from torchvision.io import read_image, ImageReadMode

data_path = Path(__file__).parent / 'data' / 'dogs_vs_cats'
data_path = data_path / 'train'
path_images = data_path.files('*.jpg')
max_w = 0
max_h = 0
for path_img in tqdm(path_images):
    img = read_image(path_img, ImageReadMode.GRAY)
    h, w = img.shape[-2:]
    max_h = max(h, max_h)
    max_w = max(w, max_w)
print(max_w)
print(max_h)