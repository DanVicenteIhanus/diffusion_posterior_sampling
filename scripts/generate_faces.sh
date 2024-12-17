#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset_dir> <dataset_name> <measurement_model> <step_size> <inpainting_noise_level>"
    exit 1
fi

MODEL_PATH="models/ffhq_baseline.pt"

process_dataset() {
    local dir="./datasets/$1"
    local dataset_name="$2"
    local measurement_model="$3"
    local step_size="$4"
    local inpainting_noise_level="$5"
    local output_dir="./output/$measurement_model/$dataset_name/"
    
    echo "================================="
    echo "Processing dataset: $dataset_name"
    echo "Input directory: $dir"
    echo "Output directory: $output_dir"
    echo "Starting DPS-sampling at $(date)"
    echo "================================="
    
    mkdir -p "$output_dir"
    
    for image_file in "$dir"/*.jpg; do
        [ -e "$image_file" ] || continue
        
        image_name=$(basename "$image_file")
        temp_dir=$(mktemp -d)
        echo ""
        echo "temp_dir: $temp_dir"
        echo "Processing image: $image_file"
        echo "Current time: $(date)"
        
        cp "$image_file" "$temp_dir/"
        
        poetry run python scripts/image_sample.py \
            --attention_resolutions "16" \
            --class_cond "False" \
            --diffusion_steps "1000" \
            --dropout "0.0" \
            --image_size "256" \
            --learn_sigma "True" \
            --noise_schedule "linear" \
            --num_channels "128" \
            --num_head_channels "64" \
            --num_res_blocks "1" \
            --resblock_updown "True" \
            --use_fp16 "False" \
            --use_scale_shift_norm "True" \
            --model_path "$MODEL_PATH" \
            --num_samples "1" \
            --batch_size "1" \
            --timestep_respacing "1000" \
            --dps_update "True" \
            --measurement_model "$measurement_model" \
            --inpainting_noise_level "$inpainting_noise_level" \
            --noise_model "poisson" \
            --sigma "0.05" \
            --step_size "$step_size" \
            --data_path "$temp_dir" \
            --sampling_batch_size "1" \
            --single_image_data "True" \
            --img_name "$image_name" \
            --output_dir "$output_dir"
            
        rm -rf "$temp_dir"
    done
}

process_dataset "$1" "$2" "$3" "$4" "$5"