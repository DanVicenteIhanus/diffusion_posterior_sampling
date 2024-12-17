#!/bin/bash

step_sizes=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)

for step_size in "${step_sizes[@]}"; do
    echo "Running experiment with step_size = $step_size"
    
    poetry run python scripts/image_sample.py \
        --attention_resolutions 32,16,8 \
        --class_cond False \
        --diffusion_steps 1000 \
        --dropout 0.0 \
        --image_size 256 \
        --learn_sigma True \
        --noise_schedule linear \
        --num_channels 256 \
        --num_head_channels 64 \
        --num_heads 4 \
        --num_res_blocks 2 \
        --resblock_updown True \
        --use_fp16 False \
        --use_scale_shift_norm True \
        --model_path models/256x256_diffusion_uncond.pt \
        --num_samples 1 \
        --batch_size 1 \
        --timestep_respacing 1000 \
        --dps_update True \
        --measurement_model Identity \
        --noise_model gaussian \
        --sigma 0.01 \
        --step_size "$step_size" \
        --img_name "img_$step_size" \
        --single_image_data True

    echo "Completed step_size = $step_size"
    echo "----------------------------------------"
done

echo "All experiments completed!"