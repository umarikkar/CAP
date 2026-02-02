
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

os.chdir(project_root)

print(os.getcwd())

import hydra
import pytest
from dataset.jumpcp import get_jumpcp_full_loader
import torch
from einops import rearrange
import h5py
from tqdm import tqdm

# Use a pipeline as a high-level helper
# Load model directly
from transformers import AutoModel
from torchvision import transforms

# import transformers

cfg_path = '/vol/research/fmodel_medical/people/umar/cha_mae_vit/configs'

# huggingface_modelpath = "recursionpharma/OpenPhenom"


# @pytest.fixture
# def huggingface_model():
#     # This step downloads the model to a local cache, takes a bit to run
#     huggingface_model = MAEModel.from_pretrained(huggingface_modelpath)
#     huggingface_model.eval()
#     return huggingface_model


# @pytest.mark.parametrize("C", [1, 4, 6, 11])
# @pytest.mark.parametrize("return_channelwise_embeddings", [True, False])
# def test_model_predict(huggingface_model, C, return_channelwise_embeddings):
#     example_input_array = torch.randint(
#         low=0,
#         high=255,
#         size=(2, C, 256, 256),
#         dtype=torch.uint8,
#         device=huggingface_model.device,
#     )
#     huggingface_model.return_channelwise_embeddings = return_channelwise_embeddings
#     embeddings = huggingface_model.predict(example_input_array)
#     expected_output_dim = 384 * C if return_channelwise_embeddings else 384
#     assert embeddings.shape == (2, expected_output_dim)




@hydra.main(version_base=None, config_path=cfg_path, config_name="main")
def main(cfg) -> None:

    cfg.logging.use_wandb = False

    resize = transforms.Resize((224, 224))

    model = AutoModel.from_pretrained("recursionpharma/OpenPhenom", trust_remote_code=True, dtype="auto")
    model.eval().cuda()


    data_loader = get_jumpcp_full_loader(**cfg.data.jumpcp, batch_size=32, eval_batch_size=32, num_workers=8, train_mode=False, do_transform=False)

    new_dset_path = "/work/um00109/CHAMMI/cha_mae_vit/frozen_features/jumpcp"
    os.makedirs(new_dset_path, exist_ok=True)

    # get joint features
    model.return_channelwise_embeddings = False
    # h5_file_path = f"{new_dset_path}/openphenom_sep.h5"
    # # cout = 1

    # # # get separate features
    # # model.return_channelwise_embeddings = True
    # # h5_file_path = f"{new_dset_path}/openphenom_joint.h5"
    # # model.tokens_per_channel=196
    # # cout = 5

    M = len(data_loader.dataset.data_path)

    # with h5py.File(h5_file_path, "w") as h5f:

    #     h5f.create_dataset("features", shape=(M, 8, 197, 384), dtype="float16", compression=None)

    #     # h5f.create_dataset("features", shape=(M, 8, 1, 384), dtype="float16", compression=None)
    #     # h5f.create_dataset("features_partial", shape=(M, 1, 1, 384), dtype="float16", compression=None)

    #     dt = h5py.string_dtype(encoding='utf-8')
    #     h5f.create_dataset("img_paths", shape=(M,), dtype=dt, compression=None)

    #     idx = 0

    #     for data in tqdm(data_loader):

    #         imgs = data['image'].cuda()
    #         img_paths = data['img_path']
    #         # imgs_partial = imgs[:, :5]

    #         B, C, *_ = imgs.shape

    #         imgs = rearrange(imgs, 'b (c n) h w -> (b c) n h w', n=1, c=8)

    #         with torch.no_grad() and torch.amp.autocast('cuda'):

    #             imgs = model.input_norm(imgs)
    #             out = model.encoder.vit_backbone.forward_features(imgs)

    #         out = out.cpu().detach().half().numpy()
    #         out = rearrange(out, '(b c) n d -> b c n d', c=C)   

    #         h5f["features"][idx:idx+B] = out
    #         h5f["img_paths"][idx:idx+B] = [os.path.basename(img_path) for img_path in img_paths]

    #         idx += B

    h5_file_path = f"{new_dset_path}/openphenom_joint.h5"

    print(h5_file_path)

    with h5py.File(h5_file_path, "w") as h5f:

        h5f.create_dataset("features", shape=(M, 1569, 384), dtype="float16", compression=None)
        h5f.create_dataset("features_partial", shape=(M, 981, 384), dtype="float16", compression=None)
        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset("img_paths", shape=(M,), dtype=dt, compression=None)

        idx = 0

        for data in tqdm(data_loader):

            imgs = data['image'].cuda()
            img_paths = data['img_path']
            imgs_partial = imgs[:, :5]

            B, C, *_ = imgs.shape

            with torch.no_grad() and torch.amp.autocast('cuda'):

                imgs = model.input_norm(imgs)
                out = model.encoder.vit_backbone.forward_features(imgs).cpu().detach()

                imgs_partial = model.input_norm(imgs_partial)
                out_partial = model.encoder.vit_backbone.forward_features(imgs_partial).cpu().detach()

            h5f["features"][idx:idx+B] = out.half().numpy()
            h5f["features_partial"][idx:idx+B] = out_partial.half().numpy()
            h5f["img_paths"][idx:idx+B] = [os.path.basename(img_path) for img_path in img_paths]
            

            idx += B




if __name__ == "__main__":
    main()
