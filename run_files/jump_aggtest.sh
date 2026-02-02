#!/bin/bash
# REMOVE set -e because it stops the loop after the first screen
# set -e

cd /vol/research/fmodel_medical/people/umar/cha_mae_vit

source /vol/research/fmodel_medical/people/umar/miniconda3/bin/activate chamaevit

# learning_rates=(2e-3 4e-3 5e-3)

              # ++train.encoder_cpt=hpa-sc \
              # ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/jumpcp/jumpcp_pretrain_checkpoint_$inp.h5 \


inp_set=(sep)
agg_set=(sep)

aggregator="protobin2"
# aggregator="abmilp"
# aggregator="ep"
# aggregator="mab"
# aggregator="mhca"

weight_decays=(1e-2)

learning_rates=(3e-3)


dataset="jumpcp"
GPUS=(  3 4 5 6 7 0 1 2)
# GPUS=( 2 3 0 1 4 5 )
# GPUS=(7)

# SEEDS=(42)
SEEDS=(43 44 45 46)


mkdir -p training_logs/logs_ICPR/${dataset}
mkdir -p outputs_linear/"${dataset}"

i=0

for inp in "${inp_set[@]}"; do
  for agg in "${agg_set[@]}"; do

    chan_setting="${inp}_${agg}"

    for lr in "${learning_rates[@]}"; do
      for wd in "${weight_decays[@]}"; do
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
              ++model.aggregator.num_prototypes=2 \
              ++logging.output_dir=outputs_linear/$dataset \
              ++logging.project_name=JUMP_BENCHMARK_LINEAR_ICPR \
              ++logging.scc_jobid=$jobid \
              ++train.learning_rate=$lr \
              ++train.weight_decay=$wd \
              ++train.seed=$seed \
              ++data.training_dataset=$dataset \
              ++train.encoder_cpt=openphenom \
              ++train.features_path=/work/um00109/CHAMMI/cha_mae_vit/frozen_features/jumpcp/openphenom_$inp.h5 \
              > $log_file 2>&1
          "

          ((i++))

        done
      done
    done
  done
done
