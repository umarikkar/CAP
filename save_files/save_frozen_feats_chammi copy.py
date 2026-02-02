import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

os.chdir(project_root)

print(os.getcwd())

import hydra

from dataset.chammiv1 import get_chammiv1_full_loader
import torch
from einops import rearrange
import h5py
from tqdm import tqdm

cfg_path = '/vol/research/fmodel_medical/people/umar/cha_mae_vit/configs'


@hydra.main(version_base=None, config_path=cfg_path, config_name="main")
def main(cfg) -> None:


    data_loader = get_chammiv1_full_loader(**cfg.data.chammiv1, batch_size=32, eval_batch_size=32, num_workers=0, persistent_workers=False)

    for chunk in ['Allen', 'HPA', 'CP']:

            loader = data_loader[chunk]

            idx = 0

            for data in tqdm(loader):

                imgs = data['image']


                img_paths = data['img_path']



    # n_tokens = {
    #     'Allen':1 + 196 * 3,
    #     'HPA':1 + 196 * 4,
    #     'CP':1 + 196 * 5,
    # }

    # with h5py.File(h5_file_path, "w") as h5f:

    #     for chunk in ['Allen', 'HPA', 'CP']:
    #         M = len(data_loader[chunk].dataset.metadata)
    #         h5f.create_dataset(f"features_{chunk}", shape=(M, n_tokens[chunk], 384), dtype="float16", compression=None)
    #         dt = h5py.string_dtype(encoding='utf-8')
    #         h5f.create_dataset(f"img_paths_{chunk}", shape=(M,), dtype=dt, compression=None)

    #         loader = data_loader[chunk]

    #         idx = 0

    #         for data in tqdm(loader):

    #             imgs = data['image'].cuda()
    #             img_paths = data['img_path']

    #             B, C, *_ = imgs.shape

    #             with torch.no_grad() and torch.amp.autocast('cuda'):
                    
    #                 out = model(imgs, return_all_tokens=True).cpu().detach()

    #             h5f[f"features_{chunk}"][idx:idx+B] = out.half().numpy()
    #             h5f[f"img_paths_{chunk}"][idx:idx+B] = img_paths
                

    #             idx += B




if __name__ == "__main__":
    main()
