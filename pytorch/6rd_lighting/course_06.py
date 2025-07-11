import json
import cv2
import numpy as np

from torch.utils.data import Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from share import *
# from cldm.logger import ImageLogger
# from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks import ModelCheckpoint



class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/cn_train_data3/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/cn_train_data3/' + source_filename)
        target = cv2.imread('./training/cn_train_data3/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == '__main__':
    # Configs
    resume_path = './pth_1.0/control_sd15_seg.pth'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    checkpoint_callback = ModelCheckpoint(
        save_last=True,                  # 总是保存最后一个epoch的模型
        save_top_k=1,                    # 保存的最佳模型数量
        # every_n_epochs=1,                # 每个epoch结束后保存
        every_n_train_steps=2000,         # 每1000训练步保存一次
        dirpath='./saved_models',        # 保存目录
        filename='model-{epoch:02d}-{step}',  # 文件名格式包含epoch和step
        save_on_train_epoch_end=True     # 在训练epoch结束时保存，不依赖验证集
    )

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)





