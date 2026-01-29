#!/bin/bash
#SBATCH -J 4dnex_lvp_train
#SBATCH -o slurm_logs/4dnex_lvp_train_%j.out
#SBATCH -e slurm_logs/4dnex_lvp_train_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mem=512G
#SBATCH --partition=dgx-b200
#SBATCH --ntasks-per-node=4
 
source ~/.bashrc
source /vast/projects/jgu32/lab/mutian/miniconda3/etc/profile.d/conda.sh

conda activate 4dnex
cd /vast/projects/jgu32/lab/mutian/4DNeX
module load cuda/13.1.0
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/vast/projects/jgu32/lab/mutian/miniconda3/envs/4dnex/bin/accelerate launch \
  --config_file configs_acc/8gpu.yaml \
  --num_processes 4\
  finetune.py \
  --model_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers-lvp-tuned \
  --model_name wan-i2v-demb-samerope \
  --model_type wan-i2v \
  --training_type lora \
  --output_dir training/fromTuned_5e4_constant \
  --report_to tensorboard \
  --raw_data \
  --raw_metadata data/lvp/meta.json \
  --data_root data/lvp \
  --train_resolution 49x480x480 \
  --train_steps 1000 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --num_workers 0 \
  --checkpointing_steps 100 \
  --checkpointing_limit 2 \
  --use_xyz_first_frame \
  --log_data_paths \
  --log_data_paths_limit 20 \
  --xyz_loss_weight 2.0 \
  --checkpoint_validation \
  --validation_dir results/infer_inputs \
  --validation_prompts prompt.txt \
  --validation_images image.txt \
  --init_domain_embeddings_path pretrained/4dnex-lora/learnable_domain_embeddings.pt \
  --domain_embedding_scale 1.0 \
  --validation_xyz_images xyz_image.txt \
  --learning_rate 2e-4 \
  --lr_scheduler constant_with_warmup\
  --lr_warmup_steps 0

