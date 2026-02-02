#!/bin/bash
set -e

# CUDA_VISIBLE_DEVICES="0" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_regular.py model=caam_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' ++data.training_dataset=jumpcp 


# CUDA_VISIBLE_DEVICES="1" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_chammi.py model=ic_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' ++data.training_dataset=chammiv1 

# CUDA_VISIBLE_DEVICES="1" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_chammi.py model=ic_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' ++data.training_dataset=chammiv1 

# CUDA_VISIBLE_DEVICES="0" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_regular.py model=ic_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' \
#     ++data.training_dataset=jumpcp ++model.patch_size=16 ++train.encoder_cpt=jumpcp-sc \
#     ++train.encoder_path=pretrained_cpts/jumpcp_pretrain_checkpoint.pth

# CUDA_VISIBLE_DEVICES="0" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_so2sat.py model=ic_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' \
#     ++data.training_dataset=so2sat ++model.patch_size=8 ++train.encoder_cpt=so2sat-sc \
#     ++train.encoder_path=pretrained_cpts/so2sat_pretrain_checkpoint.pth

# CUDA_VISIBLE_DEVICES="0" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_chammi.py model=ic_vit \
#     ++model.encoder_pooling=cls_token ++train.seed=42 ++train.train_setting='' \
#     ++data.training_dataset=chammiv1 ++model.patch_size=16 ++train.encoder_cpt=chammi-sc \
#     ++train.encoder_path=pretrained_cpts/chammi_pretrain_checkpoint.pth


CUDA_VISIBLE_DEVICES="7" /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python save_frozen_feats_regular_openphenom.py model=ic_vit \
    ++train.train_setting='' ++data.training_dataset=jumpcp 