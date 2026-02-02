#!/bin/bash

# ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/so2sat \
# ++data.so2sat.root_dir=/scratch/UMAR/Datasets/So2Sat_LCZ42 \

# ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/so2sat \
# ++data.so2sat.root_dir=/work/um00109/Datasets/So2Sat_LCZ42 \

cd /vol/research/fmodel_medical/people/umar/cha_mae_vit

source /vol/research/fmodel_medical/people/umar/miniconda3/bin/activate chamaevit

# learning_rates=(3e-5 3e-5 1e-4 1e-4 1e-4)

# weight_decays=(1e-1 1e-2 1e-1 1e-1 1e-1)

# aggregators=("abmilp" "simpool" "ep" "mab" "mhca")

learning_rates=(1e-4)



# weight_decays=(1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1 1e-1)

# aggregators=("simpool" "abmilp" "ep" "mab" "mhca")
aggregators=("protobin2")

# aggregators=("simpool")

chan_setting="joint_joint"
GPUS=( 2 3 4 5)
# chan_setting="joint_sep"
# GPUS=(6 0 1 2 )
# chan_setting="sep_joint"
# GPUS=( 3 4 5 6)
# chan_setting="sep_sep"
# GPUS=( 2 3  0 1 )

dataset="so2sat"
# GPUS=(3 4 5 6 0 1 2)
# GPUS=( 2 3  4  5 6 0 1 )

# SEEDS=(42)
SEEDS=(43 44 45 46)
# GPUS=(  4 5  6 7 0 1 2 3)

mkdir -p training_logs/logs_ICPR/$dataset
mkdir -p outputs_linear/"${dataset}"

i=0

for idx in "${!aggregators[@]}"; do
  aggregator="${aggregators[$idx]}"
  lr="${learning_rates[$idx]}"
  # wd="${weight_decays[$idx]}"

  wd=1e-1

  for seed in "${SEEDS[@]}"; do

    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    jobid="${chan_setting}_${aggregator}_${lr}_${wd}_${seed}"
    screen_name="${jobid}_gpu${gpu}"
    log_file="training_logs/logs_ICPR/${dataset}/${jobid}.txt"

    echo "Launching $jobid on GPU $gpu"

    screen -dmS "$screen_name" bash -c "
      export CUDA_VISIBLE_DEVICES=$gpu
      /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python main.py \
        model=linear_model train=train_linear \
        ++model.proxy_loss_lambda=0.0 ++model.cross_entropy_lambda=1.0 \
        ++model.aggregator.name=$aggregator \
        ++model.aggregator.chan_setting=$chan_setting \
        ++model.aggregator.num_prototypes=3 \
        ++logging.output_dir=outputs_linear/$dataset \
        ++logging.project_name=So2Sat_BENCHMARK_LINEAR_ICPR \
        ++logging.scc_jobid=$jobid \
        ++train.learning_rate=$lr \
        ++train.lr_min=1e-6 \
        ++train.weight_decay=$wd \
        ++train.seed=$seed \
        ++data.training_dataset=$dataset \
        ++train.encoder_cpt=so2sat-sc \
        ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/so2sat \
        ++data.so2sat.root_dir=/scratch/UMAR/Datasets/So2Sat_LCZ42 \
        > $log_file 2>&1
    "

    ((i++))
  done
done


# for seed in "${SEEDS[@]}"; do
#   for lr in "${learning_rates[@]}"; do
#     for wd in "${weight_decays[@]}"; do

#       gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
#       jobid="${chan_setting}_${aggregator}_${lr}_${wd}_${seed}"
#       screen_name="${jobid}_gpu${gpu}"
#       log_file="training_logs/logs_ICPR/${dataset}/${jobid}.txt"

#       echo "Launching $jobid on GPU $gpu"

#       screen -dmS "$screen_name" bash -c "
#         export CUDA_VISIBLE_DEVICES=$gpu
#         /vol/research/fmodel_medical/people/umar/miniconda3/envs/chamaevit/bin/python main.py \
#           model=linear_model train=train_linear \
#           ++model.proxy_loss_lambda=0.0 ++model.cross_entropy_lambda=1.0 \
#           ++model.aggregator.name=$aggregator \
#           ++model.aggregator.chan_setting=$chan_setting \
#           ++model.aggregator.num_prototypes=3 \
#           ++logging.output_dir=outputs_linear/$dataset \
#           ++logging.project_name=So2Sat_BENCHMARK_LINEAR_ICPR \
#           ++logging.scc_jobid=$jobid \
#           ++train.learning_rate=$lr \
#           ++train.lr_min=1e-6 \
#           ++train.weight_decay=$wd \
#           ++train.seed=$seed \
#           ++data.training_dataset=$dataset \
#           ++train.encoder_cpt=so2sat-sc \
#           ++train.features_path=/scratch4/UMAR/Datasets/frozen_features/so2sat \
#           ++data.so2sat.root_dir=/scratch/UMAR/Datasets/So2Sat_LCZ42 \
#           > $log_file 2>&1
#       "

#       ((i++))

#     done
#   done
# done


