

from __future__ import print_function, division
import os
import torch
import pandas as pd              #用于更容易地进行csv解析
from skimage import io, transform    #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np

# 忽略警告
# import warnings
# warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


def show_landmarks(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    # 绘制散点，s=10: 散点大小 marker='.': 散点样式 c='r': 红色
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(1.0)

# 辅助功能：显示批次
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


class FaceLandmarksDataset(Dataset):
    """面部标记数据集."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

    Args:
        output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。         
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


if __name__ == '__main__':
    '''
        1、读取csv文件第65条数据，数据从0开始
    '''
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

    n = 65
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].values
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    
    '''
       2、单个数据可视化 
    '''
    # plt.figure()
    # show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
    #                 landmarks)
    # plt.show()

    '''
        3、制作数据集，可视化
    '''

    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                        root_dir='data/faces/')

    # fig = plt.figure()

    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]
    #     print(i, sample['image'].shape, sample['landmarks'].shape)

    #     # 创建 1行 4列的子图， 激活第 i+1 个子图 (子图索引从 1 开始，所以i+1就是第一个图片)
    #     ax = plt.subplot(1, 4, i + 1)
        
    #     # 子图不重叠
    #     plt.tight_layout()
        
    #     # 设置子图的标题
    #     ax.set_title('Sample #{}'.format(i))
        
    #     # 关闭子图的坐标轴
    #     ax.axis('off')
        
    #     # 显示图片
    #     show_landmarks(**sample)

    #     if i == 3:
    #         plt.show()
    #         break

    '''
        4、数据变换
    '''
    # scale = Rescale(256)
    # crop = RandomCrop(128)
    # composed = transforms.Compose([Rescale(256),
    #                             RandomCrop(224)])

    # # 在样本上应用上述的每个变换。
    # fig = plt.figure()
    # sample = face_dataset[65]
    # for i, tsfrm in enumerate([scale, crop, composed]):
    #     transformed_sample = tsfrm(sample)

    #     ax = plt.subplot(1, 3, i + 1)
    #     plt.tight_layout()
    #     ax.set_title(type(tsfrm).__name__)
    #     show_landmarks(**transformed_sample)

    '''
        5.迭代数据集
    '''
    transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                            root_dir='data/faces/',
                                            transform=transforms.Compose([
                                                Rescale(256),
                                                RandomCrop(224),
                                                ToTensor()
                                            ]))

    print(f"transformed_dataset size: {len(transformed_dataset)}") 

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['landmarks'].size())

        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['landmarks'].size())

        # 观察第4批次并停止。
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    '''
        6、torchvision 数据集
            root/ants/xxx.png
            root/ants/xxy.jpeg
            root/ants/xxz.png
            .
            .
            .
            root/bees/123.jpg
            root/bees/nsdf3.png
            root/bees/asd932_.png
    '''
    # import torch
    # from torchvision import transforms, datasets

    # data_transform = transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])
    #     ])
    # hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
    #                                         transform=data_transform)
    # dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
    #                                             batch_size=4, shuffle=True,
    #                                             num_workers=4)
