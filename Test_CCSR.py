import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
from CCSR_Net import CCSR_Net
from torch.utils.data import DataLoader, Dataset
from skimage import morphology
from PIL import Image
import imageio
import kornia

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def data_normal(X):
    min = X.min()
    max = X.max()
    X_norm = np.true_divide((X - min), (max - min))
    return X_norm


class TestDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.imageFolderDataset = image_folder_dataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset.imgs[index]
        if img0_tuple[0][14:18] == 'MFFW':
            img1_tuple = img0_tuple[0][0:20] + '2' + img0_tuple[0][21:-5] + '2.tif'  # MFFW
        elif img0_tuple[0][14:19] == 'Lytro':
            img1_tuple = img0_tuple[0][0:21] + '2' + img0_tuple[0][22:-5] + '2.tif'  # Lytro
        else:
            img1_tuple = img0_tuple[0][0:23] + '2' + img0_tuple[0][24:-5] + '2.tif'  # Natural

        filename = img1_tuple.split('/')[-1]
        # filename = img1_tuple.split('\\')[-1]
        img0 = Image.open(img0_tuple[0]).convert('L')
        img1 = Image.open(img1_tuple).convert('L')
        img3 = Image.open(img0_tuple[0]).convert('RGB')
        img4 = Image.open(img1_tuple).convert('RGB')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        return img0, img1, filename, img3, img4

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def test(dataset_test):
    print("===> Loading testset")
    batch_size = 1
    transform = transforms.ToTensor()
    test_set = TestDataset(image_folder_dataset=dataset_test, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    area_ratio = 0.01
    model = CCSR_Net()
    model = model.cuda()

    print('models/CCSR' + '.pth')
    state = torch.load('models/CCSR' + '.pth')
    model.load_state_dict(state['model'])
    model.eval()

    with torch.no_grad():

        for batch, (x1, y1, fn, x, y) in enumerate(test_loader):
            x1 = x1.float()
            y1 = y1.float()

            x1 = x1.cuda()
            y1 = y1.cuda()

            x = x.float()
            y = y.float()

            x = x.cuda()
            y = y.cuda()

            d = model(x, y)
            # d = model(x1, y1)  # gray
            d = torch.sigmoid(d)
            d = d.cpu().numpy()
            d = np.transpose(d, (0, 2, 3, 1))
            d = np.squeeze(d)
            d_initial = d
            d[d > 0.5] = 1
            d[d <= 0.5] = 0
            d_binary = d

            #Small region removal
            h, w = d.shape
            d = morphology.remove_small_holes(d == 0, area_ratio * h * w)
            d = np.where(d, 0, 1)
            d = morphology.remove_small_holes(d == 1, area_ratio * h * w)
            d = np.where(d, 1, 0)
            d_final = d

            x = x.cpu().numpy()
            x = np.transpose(x, (0, 2, 3, 1))
            x = np.squeeze(x)
            y = y.cpu().numpy()
            y = np.transpose(y, (0, 2, 3, 1))
            y = np.squeeze(y)
            x1 = x1.cpu().numpy()
            x1 = np.transpose(x1, (0, 2, 3, 1))
            x1 = np.squeeze(x1)
            y1 = y1.cpu().numpy()
            y1 = np.transpose(y1, (0, 2, 3, 1))
            y1 = np.squeeze(y1)

            # w = x1 * d + y1 * (1 - d) #gray
            d = np.expand_dims(d, axis=2)
            w = x * d + y * (1 - d)

            # d_initial = (d_initial * 255).astype('uint8')
            # d_binary = (d_binary * 255).astype('uint8')
            w = (w * 255).astype('uint8')
            d_final = (d_final * 255).astype('uint8')
            if not os.path.exists("results/CCSR-Net"):
                os.makedirs("results/CCSR-Net/")
            imageio.imwrite('./results/CCSR-Net/' + fn[0][:-6] + '_CCSR.tif', w)
            # imageio.imwrite('./results/CCSR-Net/' + fn[0][:-6] + '_map.tif', d_final)
            # imageio.imwrite('results/CCSR-Net' + fn[0][:-6] + '_i.tif', d_initial)
            # imageio.imwrite('results/CCSR-Net' + fn[0][:-6] + '_b.tif', d_binary)
        print('===> Finished Testing!')


if __name__ == '__main__':
    dataset_test = torchvision.datasets.ImageFolder(root="./mydata/test/Lytro/s1")  # Lytro
    # dataset_test = torchvision.datasets.ImageFolder(root="./mydata/test/MFFW/s1")  # MFFW
    # dataset_test = torchvision.datasets.ImageFolder(root="./mydata/test/Natural/s1")  # Natural
    test(dataset_test)
