#!/bin/bash

REFERENCE_DATASET_PATH="datasets/eval_imgs_celeba_hq_sorted"
LOG_FILE="output/celeba_metrics_log.txt"
MEASUREMENT_FOLDERS=('RandomInpainting_celeba_sorted' 'BoxInpainting_celeba_sorted' \
                     'SuperRes_celeba_sorted' 'MotionBlur_celeba_sorted' \
                     'GaussianBlur_celeba_sorted')


for folder in "${MEASUREMENT_FOLDERS[@]}"; do

    echo "Measurement: $folder" >> "$LOG_FILE"

    echo "================================="
    echo "Calculating FID for: $folder"
    echo "================================="

    # Define the path to dataset1
    DATASET1_PATH="datasets/generated_imgs_celeba/$folder"

    # FID calculation
    poetry run python -m pytorch_fid "$REFERENCE_DATASET_PATH" \
                                    "$DATASET1_PATH" \
                                    | grep "FID:" >> "$LOG_FILE"

    echo "================================="
    echo "Calculating LPIPS for: $folder"
    echo "================================="

    # LPIPS calculation
    poetry run python scripts/lpips_2dirs.py -d0 "$DATASET1_PATH" \
                                            -d1 "$REFERENCE_DATASET_PATH" -o "output/lpips_dist_$folder.txt" \
                                            | grep "LPIPS:" >> "$LOG_FILE"

    echo "================================="
    echo "Calculating SSIM and PSNR for: $folder"
    echo "================================="

    # SSIM and PSNR calculation
    poetry run python scripts/ssim_psnr.py --reference_folder "$REFERENCE_DATASET_PATH" \
                                        --reconstructed_folder "$DATASET1_PATH" \
                                        | tee >(grep "SSIM:" >> "$LOG_FILE") \
                                        | grep "PSNR:" >> "$LOG_FILE"


    if [ $? -ne 0 ]; then
        echo "Error occurred while calculating FID for $folder"
        exit 1
    else
        echo "Calculation completed for $folder"
    fi
done

echo "All calculations completed successfully!"
