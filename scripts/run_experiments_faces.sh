#!/bin/bash

# In this script we run the experiments used in the report.
# The usage is described below, just run the script with:
# $ ./scripts/run_experiments_faces.sh <ExperimentType>

print_usage() {
    echo "Usage: $0 <ExperimentType>"
    echo "Available experiments:"
    echo "  BoxInpainting     - Run BoxInpainting experiment (step size: 0.5)"
    echo "  GaussianBlur      - Run GaussianBlur experiment (step size: 0.3)"
    echo "  Magnitude         - Run Magnitude experiment (step size: 0.4)"
    echo "  MotionBlur        - Run MotionBlur experiment (step size: 0.15)"
    echo "  NonlinearBlur     - Run NonlinearBlur experiment (step size: 0.3)"
    echo "  PhaseRetrieval    - Run PhaseRetrieval experiment (step size: 0.2)"
    echo "  RandomInpainting  - Run RandomInpainting experiment (step size: 40.0)"
    echo "  SuperResolution   - Run SuperResolution experiment (step size: 1.2)"
    exit 1
}

case $1 in
    "BoxInpainting")
        MEASUREMENT_MODEL="BoxInpainting"
        STEP_SIZE="0.5"
        ;;
    "GaussianBlur")
        MEASUREMENT_MODEL="GaussianBlur"
        STEP_SIZE="0.3"
        ;;
    "Magnitude")
        MEASUREMENT_MODEL="Magnitude"
        STEP_SIZE="0.4"
        ;;
    "MotionBlur")
        MEASUREMENT_MODEL="MotionBlur"
        STEP_SIZE="0.15"
        ;;
    "NonlinearBlur")
        MEASUREMENT_MODEL="NonlinearBlur"
        STEP_SIZE="0.3" 
        ;;
    "PhaseRetrieval")
        MEASUREMENT_MODEL="PhaseRetrieval"
        STEP_SIZE="0.2"
        ;;
    "RandomInpainting")
        MEASUREMENT_MODEL="RandomInpainting"
        STEP_SIZE="40."
        ;;
    "SuperResolution")
        MEASUREMENT_MODEL="SuperResolution"
        STEP_SIZE="1.2"
        ;;
    *)
        echo "Error: Unknown experiment type '$1'"
        print_usage
        ;;
esac

DATASETS=("eval_imgs_celeba_hq" "eval_imgs_ffhq")
DATASET_NAMES=("celebA" "ffhq")
INPAINTING_NOISE_LEVEL="0.92"

echo "Running $MEASUREMENT_MODEL experiment with step size $STEP_SIZE"
START_TIME=$(date +%s)
for i in "${!DATASETS[@]}"; do
    echo "Processing dataset: ${DATASET_NAMES[i]}"
    bash ./scripts/generate_faces.sh "${DATASETS[i]}" "${DATASET_NAMES[i]}" "$MEASUREMENT_MODEL" "$STEP_SIZE" "$INPAINTING_NOISE_LEVEL"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
echo "Total elapsed time = ${HOURS} hours, ${MINUTES} minutes and ${SECONDS} seconds"