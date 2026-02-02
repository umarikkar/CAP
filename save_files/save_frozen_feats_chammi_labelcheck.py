import hydra
from trainer import Trainer

from dataset.chammiv1 import get_chammiv1_full_loader
import torch
from einops import rearrange
import h5py
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg) -> None:

    cfg.logging.use_wandb = False

    data_loader = get_chammiv1_full_loader(**cfg.data.chammiv1, batch_size=32, eval_batch_size=32, num_workers=8, training_chunk=None)

    new_dset_path = "frozen_features/chammiv1"
    os.makedirs(new_dset_path, exist_ok=True)

    encoder_name = cfg.train.encoder_path.split('/')[-1].split('.')[0]
    # h5_file_path = f"{new_dset_path}/{encoder_name}_sep.h5"
    h5_file_path = f"{new_dset_path}/{encoder_name}_sep.h5"

    n_chans = {
        'Allen':3,
        'HPA':4,
        'CP':5,
    }

    for chunk in ['Allen', 'HPA', 'CP']:
        M = len(data_loader[chunk].dataset.metadata)

        folderName = f'assets/single_channel_images/v2/imgs_{chunk}'
        os.makedirs(folderName, exist_ok=True)

        loader = data_loader[chunk]

        idx = 0

        label_counts = {}

        for data in tqdm(loader):

            imgs = data['image']
            img_paths = data['img_path']
            label = data['label']

            B = len(imgs)

            for im_idx, (img, label_val) in enumerate(zip(imgs, label)):
                label_int = int(label_val)
                # Initialize count for the label if not present
                if label_int not in label_counts:
                    label_counts[label_int] = 0

                # Skip saving if 50 images for this label have been saved
                if label_counts[label_int] >= 10:
                    continue

                img_np = img.detach().cpu().numpy()
                # plt.figure(figsize=(2 * len(img_np), 2.2))

                for ii, im in enumerate(img_np):
                    im = (im - im.min()) / (im.max() - im.min())
                    # plt.subplot(1, len(img_np), ii + 1)
                    # plt.imshow(im)
                    # plt.axis('off')
                    plt.imsave(os.path.join(folderName, f'label_{label_int}_idx_{label_counts[label_int]}_channel_{ii}.png'), im)
                
                # plt.suptitle(label_val)
                # plt.savefig(os.path.join(folderName, f'label_{label_int}_idx_{label_counts[label_int]}'))
                # plt.close()

                label_counts[label_int] += 1

            idx += B

            # if idx > 500:
            #     break





if __name__ == "__main__":
    main()
