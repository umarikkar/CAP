import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

os.chdir(project_root)

print(os.getcwd())

import hydra
from trainer import Trainer

from dataset.chammiv1 import get_chammiv1_full_loader
import torch
from einops import rearrange
import h5py
from tqdm import tqdm

cfg_path = '/vol/research/fmodel_medical/people/umar/cha_mae_vit/configs'

from transformers import AutoModel
from torchvision import transforms

@hydra.main(version_base=None, config_path=cfg_path, config_name="main")
def main(cfg) -> None:

    resize = transforms.Resize((224, 224))

    cfg.logging.use_wandb = False

    model = AutoModel.from_pretrained("recursionpharma/OpenPhenom", trust_remote_code=True, dtype="auto")
    model.eval().cuda()

    data_loader = get_chammiv1_full_loader(**cfg.data.chammiv1, batch_size=32, eval_batch_size=32, num_workers=8, do_transform=False)

    new_dset_path = "/work/um00109/CHAMMI/cha_mae_vit/frozen_features/chammiv1"
    os.makedirs(new_dset_path, exist_ok=True)

    # # get joint features
    # model.return_channelwise_embeddings = False
    # h5_file_path = f"{new_dset_path}/openphenom_joint_joint.h5"
    # cout = 1

    # get joint features
    model.return_channelwise_embeddings = False
    h5_file_path = f"{new_dset_path}/openphenom_sep.h5"
    cout = 1

    # # # get separate features
    # model.return_channelwise_embeddings = False
    # h5_file_path = f"{new_dset_path}/openphenom_joint_sep_cls.h5"
    # model.tokens_per_channel=196
    # # cout = 5

    n_chans = {
        'Allen':3,
        'HPA':4,
        'CP':5,
    }

    with h5py.File(h5_file_path, "w") as h5f:

        for chunk in ['Allen', 'HPA', 'CP']:
            M = len(data_loader[chunk].dataset.metadata)
            h5f.create_dataset(f"features_{chunk}", shape=(M, n_chans[chunk], 197, 384), dtype="float16", compression=None)
            dt = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset(f"img_paths_{chunk}", shape=(M,), dtype=dt, compression=None)

            loader = data_loader[chunk]

            idx = 0

            for data in tqdm(loader):

                imgs = data['image'].cuda()
                img_paths = data['img_path']

                imgs = (resize(imgs) * 255).to(torch.uint8)
                
                B, C, *_ = imgs.shape

                imgs = rearrange(imgs, 'b (c 1) h w -> (b c) 1 h w', c=C)

                with torch.no_grad() and torch.amp.autocast('cuda'):

                    imgs = model.input_norm(imgs)
                    out = model.encoder.vit_backbone.forward_features(imgs)

                out = out.cpu().detach().half().numpy()
                out = rearrange(out, '(b c) n d -> b c n d', c=C)

                h5f[f"features_{chunk}"][idx:idx+B] = out
                h5f[f"img_paths_{chunk}"][idx:idx+B] = img_paths

                idx += B

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

    #             imgs = (resize(imgs) * 255).to(torch.uint8)

    #             B, C, *_ = imgs.shape

    #             with torch.no_grad() and torch.amp.autocast('cuda'):
                    
    #                 imgs = model.input_norm(imgs)
    #                 out = model.encoder.vit_backbone.forward_features(imgs)

    #             out = out.cpu().detach().half().numpy()

    #             h5f[f"features_{chunk}"][idx:idx+B] = out
    #             h5f[f"img_paths_{chunk}"][idx:idx+B] = img_paths
                

    #             idx += B




if __name__ == "__main__":
    main()
