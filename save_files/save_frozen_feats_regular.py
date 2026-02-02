
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

os.chdir(project_root)

print(os.getcwd())

import hydra
from trainer import Trainer
from dataset.jumpcp import get_jumpcp_full_loader
import torch
from einops import rearrange
import h5py
from tqdm import tqdm

cfg_path = '/vol/research/fmodel_medical/people/umar/cha_mae_vit/configs'


@hydra.main(version_base=None, config_path=cfg_path, config_name="main")
def main(cfg) -> None:

    cfg.logging.use_wandb = False

    trainer = Trainer(cfg)
    model = trainer.model.eval()

    data_loader = get_jumpcp_full_loader(**cfg.data.jumpcp, batch_size=32, eval_batch_size=32, num_workers=8, train_mode=False)


    new_dset_path = "/work/um00109/CHAMMI/cha_mae_vit/frozen_features/jumpcp"
    os.makedirs(new_dset_path, exist_ok=True)

    encoder_name = cfg.train.encoder_path.split('/')[-1].split('.')[0]
    M = len(data_loader.dataset.data_path)

    
    # h5_file_path = f"{new_dset_path}/{encoder_name}_sep.h5"

    

    # with h5py.File(h5_file_path, "w") as h5f:

    #     h5f.create_dataset("features", shape=(M, 8, 197, 384), dtype="float16", compression=None)

    #     # h5f.create_dataset("features", shape=(M, 8, 1, 384), dtype="float16", compression=None)

    #     dt = h5py.string_dtype(encoding='utf-8')
    #     h5f.create_dataset("img_paths", shape=(M,), dtype=dt, compression=None)

    #     idx = 0

    #     for idx2, data in enumerate(tqdm(data_loader)):

    #         imgs = data['image'].cuda()
    #         img_paths = data['img_path']
    #         # imgs_partial = imgs[:, :5]

    #         B, C, *_ = imgs.shape

    #         imgs = rearrange(imgs, 'B (C 1) ... -> (B C) 1 ...', B=B, C=C)

    #         with torch.no_grad() and torch.amp.autocast('cuda'):
                
    #             out = model(imgs, return_all_tokens=True).cpu().detach()
    #             # out_partial = model(imgs_partial, return_all_tokens=True).cpu().detach()

    #             # extract CLS token only. comment out this bit when we run the overall evaluations.
    #             # out = out[:, :1]
    #             # out_partial = out_partial[:, :1]

    #         out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()
    #         # out_partial = rearrange(out_partial, '(B C) N D -> B C N D', B=B, C=5).half().numpy()

    #         h5f["features"][idx:idx+B] = out
    #         h5f["img_paths"][idx:idx+B] = [os.path.basename(img_path) for img_path in img_paths]

    #         idx += B


    # h5_file_path = f"{new_dset_path}/{encoder_name}_joint.h5"

    # with h5py.File(h5_file_path, "w") as h5f:

    #     h5f.create_dataset("features", shape=(M, 1569, 384), dtype="float16", compression=None)
    #     h5f.create_dataset("features_partial", shape=(M, 981, 384), dtype="float16", compression=None)
    #     dt = h5py.string_dtype(encoding='utf-8')
    #     h5f.create_dataset("img_paths", shape=(M,), dtype=dt, compression=None)

    #     idx = 0

    #     for data in tqdm(data_loader):

    #         imgs = data['image'].cuda()
    #         img_paths = data['img_path']
    #         imgs_partial = imgs[:, :5]

    #         B, C, *_ = imgs.shape

    #         with torch.no_grad() and torch.amp.autocast('cuda'):
                
    #             out = model(imgs, return_all_tokens=True).cpu().detach()
    #             out_partial = model(imgs_partial, return_all_tokens=True).cpu().detach()

    #         # out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()
    #         # out_partial = rearrange(out_partial, '(B C) N D -> B C N D', B=B, C=5).half().numpy()

    #         h5f["features"][idx:idx+B] = out.half().numpy()
    #         h5f["features_partial"][idx:idx+B] = out_partial.half().numpy()
    #         h5f["img_paths"][idx:idx+B] = [os.path.basename(img_path) for img_path in img_paths]
            

    #         idx += B

    h5_file_path = f"{new_dset_path}/{encoder_name}_cls.h5"

    with h5py.File(h5_file_path, "w") as h5f:

        h5f.create_dataset("features", shape=(M, 1, 384), dtype="float16", compression=None)
        h5f.create_dataset("features_partial", shape=(M, 1, 384), dtype="float16", compression=None)
        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset("img_paths", shape=(M,), dtype=dt, compression=None)

        idx = 0

        for data in tqdm(data_loader):

            imgs = data['image'].cuda()
            img_paths = data['img_path']
            imgs_partial = imgs[:, :5]

            B, C, *_ = imgs.shape

            with torch.no_grad() and torch.amp.autocast('cuda'):
                
                out = model(imgs, return_all_tokens=True).cpu().detach()[:,:1]
                out_partial = model(imgs_partial, return_all_tokens=True).cpu().detach()[:,:1]

            # out = rearrange(out, '(B C) N D -> B C N D', B=B, C=C).half().numpy()
            # out_partial = rearrange(out_partial, '(B C) N D -> B C N D', B=B, C=5).half().numpy()

            h5f["features"][idx:idx+B] = out.half().numpy()
            h5f["features_partial"][idx:idx+B] = out_partial.half().numpy()
            h5f["img_paths"][idx:idx+B] = [os.path.basename(img_path) for img_path in img_paths]
            

            idx += B




if __name__ == "__main__":
    main()
