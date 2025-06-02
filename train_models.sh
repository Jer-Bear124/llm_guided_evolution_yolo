#!/bin/bash
#SBATCH --job-name=train_network_file # Job name
#SBATCH -t 8:00:00                    # Runtime in D-HH:MM
#SBATCH --mem 256G                    # Memory request
#SBATCH -c 4                          # number of CPU cores
#SBATCH --gres=gpu:2                  # Request 2 GPUs
#SBATCH -C "H200"                     # Request specific GPU type (H200)

# --- Configuration ---
# Set RESUME_TRAINING to true to resume, false to start new.
RESUME_TRAINING=true # <--- SET THIS TO true or false

# Path to the model config YAML (used if RESUME_TRAINING=false)
MODEL_CONFIG_YAML="/home/hice1/yzhang3942/scratch/llm-guided-evolution/sota/ultralytics/ultralytics/cfg/models/llm/network_xXxdykdqEciHfSnuAwLOCjgT3hk.yaml"

# Path to the checkpoint file (used if RESUME_TRAINING=true)
# IMPORTANT: Update this path to the specific 'last.pt' file from the run you want to resume.
RESUME_CHECKPOINT_PT="/home/hice1/yzhang3942/scratch/llm-guided-evolution/runs/detect/train7/weights/last.pt" # <--- UPDATE THIS IF RESUMING

# Path to the data config YAML
DATA_CONFIG_YAML="sota/ultralytics/coco.yaml"

# Path to the python script
# IMPORTANT: Ensure this python script accepts '--model' and '--resume' arguments
PYTHON_SCRIPT="train_network.py"

# Total number of epochs desired for the run
TOTAL_EPOCHS=100

# --- Set Model Path and Resume Flag based on Configuration ---
MODEL_ARG=""
RESUME_FLAG=""

if [ "$RESUME_TRAINING" = true ]; then
    echo "Configuration: Resuming training."
    MODEL_ARG="$RESUME_CHECKPOINT_PT"
    RESUME_FLAG="--resume"
    if [ ! -f "$RESUME_CHECKPOINT_PT" ]; then
        echo "Error: Resume checkpoint file not found at $RESUME_CHECKPOINT_PT"
        exit 1
    fi
else
    echo "Configuration: Starting new training."
    MODEL_ARG="$MODEL_CONFIG_YAML"
    RESUME_FLAG="" # No resume flag when starting new
    if [ ! -f "$MODEL_CONFIG_YAML" ]; then
        # Allow starting from standard model names like yolov8n.pt
        if [[ ! "$MODEL_CONFIG_YAML" == *.pt && ! "$MODEL_CONFIG_YAML" == *.yaml ]]; then
             echo "Warning: MODEL_CONFIG_YAML '$MODEL_CONFIG_YAML' is not a .yaml or .pt file. Assuming it's a standard model name."
        elif [ ! -f "$MODEL_CONFIG_YAML" ]; then
             echo "Error: Model config/weights file not found at $MODEL_CONFIG_YAML"
             exit 1
        fi
    fi
fi


# --- Environment Setup ---
echo "resuming training of network(to 100 epochs)" # User's message
hostname
module load cuda  # Load necessary modules
module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh # Source conda
conda activate llm_env # Activate environment
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'llm_env'"
    exit 1
fi
conda info # Print environment info
which python # Show python path
echo "Environment setup CHECK DONE"


# --- File Checks ---
# Check relative paths based on the assumption that the script is run from the parent directory.
CURRENT_DIR=$(pwd) # Get the directory where sbatch was run
EXPECTED_DATA_PATH="$CURRENT_DIR/$DATA_CONFIG_YAML"
EXPECTED_SCRIPT_PATH="$CURRENT_DIR/$PYTHON_SCRIPT"

# Basic check for data config existence (relative path assumed)
if [[ ! "$DATA_CONFIG_YAML" == *.yaml ]]; then
    echo "Warning: DATA_CONFIG_YAML does not end with .yaml. Assuming it's a dataset name like 'coco.yaml'."
elif [ ! -f "$EXPECTED_DATA_PATH" ]; then
    echo "Warning: Data config file not found at expected relative path: $EXPECTED_DATA_PATH"
    echo "Ensure you are running sbatch from the correct directory,"
    echo "or that '$DATA_CONFIG_YAML' specifies correct absolute paths or downloadable dataset name."
fi

if [ ! -f "$EXPECTED_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at expected relative path: $EXPECTED_SCRIPT_PATH"
    echo "Ensure you are running sbatch from the correct directory."
    exit 1
fi


# --- Run the Python Training Script ---
echo "Starting Python training script: $PYTHON_SCRIPT"
echo "Using Model Argument: $MODEL_ARG"
echo "Using Data Config: $DATA_CONFIG_YAML"
echo "Target Epochs: $TOTAL_EPOCHS"
echo "Resume Flag: $RESUME_FLAG"
echo "Resuming network, 100 epochs!"

# Execute the python script using the activated environment's python.
# NOTE: Assumes train_network.py uses --model and --resume arguments now.
python -u "$PYTHON_SCRIPT" \
    --model "$MODEL_ARG" \
    --data_config "$DATA_CONFIG_YAML" \
    --epochs "$TOTAL_EPOCHS" \
    $RESUME_FLAG # This will be empty if not resuming

# Capture exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Python script failed with exit code $EXIT_CODE"
fi

echo "Training script finished."

# --- Optional: Deactivate environment ---
# conda deactivate

echo "Job completed with exit code $EXIT_CODE."
exit $EXIT_CODE