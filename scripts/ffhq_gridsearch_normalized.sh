#!/bin/bash

DIR="./datasets/celeba_hq_search"
MODEL_PATH="models/ffhq_baseline.pt"
step_sizes=(1.0 2.0 3.0 4.0 5.0 7.5 10.0)
START_TIME=$(date +%s)

echo "Starting experiments at $(date)"

for image_file in "$DIR"/*.jpg; do
    [ -e "$image_file" ] || continue
    
    echo "Processing image: $image_file"
    
    for step_size in "${step_sizes[@]}"; do
        echo "Running experiment with step_size = $step_size"
        echo "Current time: $(date)"
        temp_dir=$(mktemp -d)
        cp "$image_file" "$temp_dir/"
        
        poetry run python scripts/image_sample.py \
            --attention_resolutions "16" \
            --class_cond False \
            --diffusion_steps 1000 \
            --dropout 0.0 \
            --image_size 256 \
            --learn_sigma True \
            --noise_schedule linear \
            --num_channels 128 \
            --num_head_channels 64 \
            --num_res_blocks 1 \
            --resblock_updown True \
            --use_fp16 False \
            --use_scale_shift_norm True \
            --model_path "$MODEL_PATH" \
            --num_samples 1 \
            --batch_size 1 \
            --timestep_respacing 1000 \
            --dps_update True \
            --measurement_model Grayscale \
            --inpainting_noise_level 0.92 \
            --noise_model gaussian \
            --sigma 0.05 \
            --step_size "$step_size" \
            --data_path "$temp_dir" \
            --sampling_batch_size 1 \
            --single_image_data True \
            --output_dir "./output/Grayscale/celebA_grid/$step_size"
        rm -rf "$temp_dir"
        
        echo "Completed step_size = $step_size for image $image_file"
        echo "----------------------------------------"
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "All experiments completed!"
echo "Total elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"