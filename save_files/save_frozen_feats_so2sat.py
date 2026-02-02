
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

os.chdir(project_root)

print(os.getcwd())

import hydra
from trainer import Trainer

from dataset.jumpcp import get_jumpcp_full_loader
from dataset.so2sat import get_so2sat_full_loader
import torch
from einops import rearrange
import h5py
import os
from tqdm import tqdm


cfg_path = '/vol/research/fmodel_medical/people/umar/cha_mae_vit/configs'


@hydra.main(version_base=None, config_path=cfg_path, config_name="main")
def main(cfg) -> None:

    cfg.logging.use_wandb = False

    trainer = Trainer(cfg)
    model = trainer.model.eval()

    train_loader, val_loader, test_loader = get_so2sat_full_loader(**cfg.data.so2sat, batch_size=16, eval_batch_size=32, num_workers=8, train_mode=False)

    new_dset_path = "/work/um00109/CHAMMI/cha_mae_vit/frozen_features/so2sat"
    # new_dset_path = "frozen_features/so2sat"
    os.makedirs(new_dset_path, exist_ok=True)

    # for data_loader, split in zip([train_loader, val_loader, test_loader], ['train', 'valid', 'test']):
    for data_loader, split in zip([ val_loader, test_loader], ['valid', 'test']):

        # h5_file_path = f"{new_dset_path}/so2sat_{split}_sep.h5"

        # M = len(data_loader.dataset.file['label'])

        # with h5py.File(h5_file_path, "w") as h5f:

        #     h5f.create_dataset("features", shape=(M, 18, 17, 384), dtype="float16", compression=None)
        #     h5f.create_dataset("label", shape=(M), dtype="int", compression=None)

        #     idx = 0

        #     for data in tqdm(data_loader):

        #         imgs = data['image'].cuda()
        #         img_paths = data['label']

        #         B, C, *_ = imgs.shape

        #         imgs = rearrange(imgs, 'B (C 1) ... -> (B C) 1 ...', B=B, C=C)

        #         with torch.no_grad() and torch.amp.autocast('cuda'):
                    
        #             out = model(imgs, return_all_tokens=True).cpu().detach()

        #         out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()

        #         h5f["features"][idx:idx+B] = out
        #         h5f["label"][idx:idx+B] = img_paths

        #         idx += B


        # h5_file_path = f"{new_dset_path}/so2sat_{split}_joint.h5"

        # M = len(data_loader.dataset.file['label'])

        # with h5py.File(h5_file_path, "w") as h5f:

        #     h5f.create_dataset("features", shape=(M, 289, 384), dtype="float16", compression=None)
        #     h5f.create_dataset("features_partial", shape=(M, 129, 384), dtype="float16", compression=None)
        #     h5f.create_dataset("label", shape=(M), dtype="int", compression=None)

        #     idx = 0

        #     for data in tqdm(data_loader):

        #         imgs = data['image'].cuda()
        #         img_paths = data['label']
        #         imgs_partial = imgs[:, :8]

        #         B, C, *_ = imgs.shape

        #         with torch.no_grad() and torch.amp.autocast('cuda'):
                    
        #             out = model(imgs, return_all_tokens=True).cpu().detach()
        #             out_partial = model(imgs_partial, return_all_tokens=True).cpu().detach()

        #         # out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()
        #         # out_partial = rearrange(out_partial, '(B C) N D -> B C N D', B=B, C=5).half().numpy()

        #         h5f["features"][idx:idx+B] = out.half().numpy()
        #         h5f["features_partial"][idx:idx+B] = out_partial.half().numpy()
        #         h5f["label"][idx:idx+B] = img_paths

        #         idx += B


        h5_file_path = f"{new_dset_path}/so2sat_{split}_cls.h5"

        M = len(data_loader.dataset.file['label'])

        with h5py.File(h5_file_path, "w") as h5f:

            h5f.create_dataset("features", shape=(M, 1, 384), dtype="float16", compression=None)
            h5f.create_dataset("features_partial", shape=(M, 1, 384), dtype="float16", compression=None)
            h5f.create_dataset("label", shape=(M), dtype="int", compression=None)

            idx = 0

            for data in tqdm(data_loader):

                imgs = data['image'].cuda()
                img_paths = data['label']
                imgs_partial = imgs[:, :8]

                B, C, *_ = imgs.shape

                with torch.no_grad() and torch.amp.autocast('cuda'):
                    
                    out = model(imgs, return_all_tokens=True).cpu().detach()[:,:1]
                    out_partial = model(imgs_partial, return_all_tokens=True).cpu().detach()[:,:1]

                # out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()
                # out_partial = rearrange(out_partial, '(B C) N D -> B C N D', B=B, C=5).half().numpy()

                h5f["features"][idx:idx+B] = out.half().numpy()
                h5f["features_partial"][idx:idx+B] = out_partial.half().numpy()
                h5f["label"][idx:idx+B] = img_paths

                idx += B







if __name__ == "__main__":
    main()
