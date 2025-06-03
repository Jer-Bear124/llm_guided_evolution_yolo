# train_network.py
import argparse
from ultralytics import YOLO
import torch # PyTorch is a dependency for ultralytics
import os

def train_model(model_path, data_config_path, epochs, resume_training):
    """
    Trains or resumes training for a YOLO model.

    Args:
        model_path (str): Path to the model configuration YAML file (e.g., 'yolov8n.yaml')
                          OR path to a checkpoint file (e.g., 'path/to/runs/detect/exp/weights/last.pt')
                          for resuming.
        data_config_path (str): Path to the dataset configuration YAML file.
        epochs (int): Total number of training epochs desired.
        resume_training (bool): Flag indicating whether to resume training. If True, model_path
                                should point to a '.pt' checkpoint file.
    """
    print(f"Starting YOLO model training...")
    if resume_training:
        print(f"Resuming training from checkpoint: {model_path}")
        if not os.path.exists(model_path) or not model_path.endswith('.pt'):
             print(f"Error: Resume path '{model_path}' does not exist or is not a .pt file.")
             return # Exit if resume path is invalid
    else:
        print(f"Starting new training from config: {model_path}")
        if not os.path.exists(model_path) or not model_path.endswith('.yaml'):
             print(f"Error: Model config path '{model_path}' does not exist or is not a .yaml file.")
             return # Exit if config path is invalid

    print(f"Data Configuration: {data_config_path}")
    print(f"Target Total Epochs: {epochs}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("Warning: No GPU detected, training will run on CPU.")


    try:
        # Load the YOLO model.
        # If model_path is a .yaml, it initializes a new model.
        # If model_path is a .pt, it loads the weights and state from the checkpoint.
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        # Train the model
        print("Configuring model training...")
        # The 'train' method handles the complete training loop.
        # - model.train() automatically handles resuming if the model was loaded from a .pt file
        #   and the 'resume=True' flag is passed.
        # - epochs: Specifies the *total* number of epochs for the training run. If resuming,
        #           it will train only the remaining epochs.
        results = model.train(data=data_config_path,
                              epochs=epochs,
                              imgsz=640, # Example image size, adjust as needed
                              resume=resume_training, # Key flag for resuming
                              device=None # Auto-detect device (recommended)
                              )

        print("Training process finished.")
        print(f"Results saved to: {results.save_dir}") # Ultralytics saves results automatically

    except FileNotFoundError as e:
        print(f"Error: Configuration or data file not found - {e}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # You might want to add more specific error handling here
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Train or resume training for a YOLO object detection model.")
    parser.add_argument(
        "--model", # Changed from --model_config
        type=str,
        required=True,
        help="Path to the model configuration YAML file (e.g., yolov8n.yaml) for new training, "
             "OR path to the checkpoint file (e.g., path/to/last.pt) for resuming."
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data configuration YAML file (e.g., dataset.yaml)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50, # Default total epochs
        help="Total number of training epochs desired."
    )
    parser.add_argument(
        "--resume",
        action='store_true', # Makes this a flag; if present, it's True, otherwise False
        help="Flag to indicate resuming training. If set, --model must point to a .pt file."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the training function
    train_model(args.model, args.data_config, args.epochs, args.resume)

































# Previous stuff

# def train_model(model_config_path, data_config_path, epochs):
#     """
#     Trains a YOLO model using the specified configuration files and number of epochs.

#     Args:
#         model_config_path (str): Path to the model configuration YAML file (e.g., 'yolov8n.yaml').
#         data_config_path (str): Path to the dataset configuration YAML file.
#         epochs (int): Number of training epochs.
#     """
#     print(f"Starting YOLO model training...")
#     print(f"Model Configuration: {model_config_path}")
#     print(f"Data Configuration: {data_config_path}")
#     print(f"Number of Epochs: {epochs}")
#     print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
#     if torch.cuda.is_available():
#         print(f"Number of GPUs available: {torch.cuda.device_count()}")

#     try:
#         # Load the YOLO model from the specified configuration YAML file
#         # This initializes a new model architecture defined in the yaml
#         print(f"Loading model from config: {model_config_path}")
#         model = YOLO(model_config_path)

#         # Train the model
#         print("Starting model training...")
#         # The 'train' method handles the complete training loop.
#         # - data: Path to the data configuration yaml. This file specifies paths
#         #         for training/validation images and class names.
#         # - epochs: Number of complete passes through the training dataset.
#         # - imgsz: Input image size (optional, can be inferred or set).
#         # - batch: Batch size (optional, default is 16, -1 for auto batch size).
#         # - device: Specify device ('cpu', '0', '0,1' etc.). If None, uses available GPUs or CPU.
#         results = model.train(data=data_config_path,
#                               epochs=epochs,
#                               imgsz=640, # Example image size, adjust as needed
#                               device="cuda") # Auto-detect device

#         print("Training completed successfully.")
#         print(f"Results saved to: {results.save_dir}") # Ultralytics saves results automatically

#     except FileNotFoundError as e:
#         print(f"Error: Configuration file not found - {e}")
#     except Exception as e:
#         print(f"An error occurred during training: {e}")

# if __name__ == "__main__":
#     # Set up argument parser for command-line execution
#     parser = argparse.ArgumentParser(description="Train a YOLO object detection model.")
#     parser.add_argument(
#         "--model_config",
#         type=str,
#         required=True,
#         help="Path to the model configuration YAML file (e.g., yolov8n.yaml)."
#     )
#     parser.add_argument(
#         "--data_config",
#         type=str,
#         required=True,
#         help="Path to the data configuration YAML file (e.g., dataset.yaml)."
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=100, # Default to 12 epochs as requested
#         help="Number of training epochs."
#     )

#     # Parse arguments
#     args = parser.parse_args()

#     # Call the training function
#     train_model(args.model_config, args.data_config, args.epochs)
