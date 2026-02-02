#!/bin/bash

# ++train.features_path=/scratch/UMAR/Datasets/frozen_features/chammiv1/vit_s_sc_hpa_norm_mean_std_joint.h5 \

# ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/chammiv1/vit_s_sc_hpa_norm_mean_std_joint.h5 \
# ++eval.chammiv1.root_dir=/scratch/UMAR/Datasets/chammi_dataset \

# ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/chammiv1/vit_s_sc_hpa_norm_mean_std_joint.h5 \
# ++eval.chammiv1.root_dir=/scratch1/test_data/chammi_dataset \


# ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/chammiv1/chammi_pretrain_checkpoint_$inp.h5 \
# ++eval.chammiv1.root_dir=/scratch/UMAR/Datasets/chammi_dataset \
# ++train.encoder_cpt=chammi_sc \

# ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/chammiv1/chammi_pretrain_checkpoint_$inp.h5 \
# ++eval.chammiv1.root_dir=/scratch1/test_data/chammi_dataset \
# ++train.encoder_cpt=chammi_sc \

# ++train.encoder_cpt=hpa_single \ 
# ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/chammiv1/vit_s_sc_hpa_norm_mean_std_sep.h5 \
# ++eval.chammiv1.root_dir=/scratch1/test_data/chammi_dataset \


# ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/chammiv1/vit_s_sc_hpa_norm_mean_std_$inp.h5 \
# ++eval.chammiv1.root_dir=/scratch/UMAR/Datasets/chammi_dataset \
# ++train.encoder_cpt=hpa_sc \ 


cd /vol/research/fmodel_medical/people/umar/cha_mae_vit

source /vol/research/fmodel_medical/people/umar/miniconda3/bin/activate chamaevit



temperatures=(0.2)

# aggregator="simpool"
# aggregator="abmilp"
aggregator="protobin2"
# aggregator="mab"
# aggregator="mab"
# aggregator="mhca"

learning_rates=(3e-2)

inp=joint
agg=joint

chan_setting="${inp}_${agg}"

dataset="chammiv1"

# GPUS=( 4 5 6 0 1 2 3 )
# GPUS=(  7 0 2  3 4 5 6)
GPUS=(0 1 2 3)
# GPUS=(4 5 6 7)


SEEDS=(43 44 45 46)
# SEEDS=(42)

mkdir -p training_logs/logs_ICPR/${dataset}
mkdir -p outputs_linear/"${dataset}"

i=0

for seed in "${SEEDS[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for temp in "${temperatures[@]}"; do

      gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
      jobid="${chan_setting}_${aggregator}_${lr}_${temp}_seed${seed}"
      screen_name="${jobid}_gpu${gpu}"
      log_file="training_logs/logs_ICPR/${dataset}/${jobid}.txt"

      echo "Launching $jobid on GPU $gpu"

      screen -dmS "$screen_name" bash -c "
        export CUDA_VISIBLE_DEVICES=$gpu
        /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python main.py \
          model=linear_model train=train_linear \
          ++model.proxy_loss_lambda=1.0 ++model.cross_entropy_lambda=0.0 \
          ++model.aggregator.name=$aggregator \
          ++model.aggregator.num_prototypes=2 \
          ++model.proxy_temperature=$temp \
          ++model.use_proxy_head=False \
          ++model.aggregator.chan_setting=$chan_setting \
          ++logging.output_dir=outputs_linear/$dataset \
          ++logging.project_name=CHAMMI_BENCHMARK_LINEAR_ICPR \
          ++logging.scc_jobid=$jobid \
          ++train.learning_rate=$lr \
          ++train.weight_decay=1e-3 \
          ++train.seed=$seed \
          ++data.training_dataset=$dataset \
          ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/chammiv1/openphenom_$inp.h5 \
          ++eval.chammiv1.root_dir=/scratch1/test_data/chammi_dataset \
          ++train.encoder_cpt=openphenom \
          ++eval.eval_every_epochs=30 \
          > $log_file 2>&1
      "

      ((i++))

    done
  done
done

