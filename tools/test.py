#!/usr/bin/env python
"""MMDetection 3.x Inference Script

This script runs inference using a trained MMDetection model on images and can 
optionally log the model to MLflow for tracking and deployment.

Usage:
    python test.py --config CONFIG_FILE --checkpoint CHECKPOINT_FILE --input INPUT_PATH 
                  [--output OUTPUT_DIR] [--score-thr SCORE_THRESHOLD] [--device DEVICE]
                  [--log-mlflow] [--mlflow-experiment EXPERIMENT_NAME]
"""
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

import argparse
import glob
import os
import time
import torch
import re
import sys

# Check for packaging package and install if needed
try:
    from packaging import version
except ImportError:
    import subprocess
    import sys
    print("Installing packaging package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    from packaging import version

import mlflow
import mlflow.pytorch
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from mmengine.registry import VISUALIZERS
from mmengine.utils import mkdir_or_exist
import torch.serialization
from mmengine.logging.history_buffer import HistoryBuffer

# Fix for PyTorch 2.6+: Allow loading HistoryBuffer class when deserializing checkpoints
# Starting with PyTorch 2.6, the default value of weights_only in torch.load changed to True
# for security reasons, which blocks loading custom classes from checkpoints
torch.serialization.add_safe_globals({'mmengine.logging.history_buffer.HistoryBuffer': HistoryBuffer})

def get_training_env_from_log(log_file_path):
    """Parses an MMDetection training log file to extract environment info."""
    env_info = {}
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Regex patterns to find specific lines
        patterns = {
            'python_version': r"Python: ([\d\.]+)",
            'pytorch_version': r"PyTorch: ([^\s]+)",
            'torchvision_version': r"TorchVision: ([^\s]+)",
            'opencv_version': r"OpenCV: ([^\s]+)",
            'mmengine_version': r"MMEngine: ([^\s]+)",
            'cuda_version': r"CUDA Runtime (\d+\.\d+)",
            'cudnn_version': r"CuDNN (\d+\.\d+)",
            'gcc_version': r"GCC: gcc.*?([\d\.]+)",
        }

        # Search for patterns
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                env_info[key] = match.group(1)
            else:
                # Handle cases like CUDA not being available or certain lines missing
                if key == 'cuda_version' and "CUDA available: False" in content:
                    env_info[key] = "Not available"
                elif key == 'cudnn_version' and "CUDA available: False" in content:
                     env_info[key] = "Not available"


    except FileNotFoundError:
        print(f"  [Warning] Training log file not found: {log_file_path}")
    except Exception as e:
        print(f"  [Warning] Error parsing training log file {log_file_path}: {e}")
    return env_info

def get_current_env():
    """Gets current inference environment information."""
    env_info = {
        'python_version': sys.version.split(' ')[0],
        'pytorch_version': torch.__version__,  # Keep full version string
    }
    try:
        import torchvision
        env_info['torchvision_version'] = torchvision.__version__  # Keep full version string
    except ImportError:
        env_info['torchvision_version'] = "Not installed"
    try:
        import cv2
        env_info['opencv_version'] = cv2.__version__
    except ImportError:
        env_info['opencv_version'] = "Not installed"
    try:
        import mmengine
        env_info['mmengine_version'] = mmengine.__version__
    except ImportError:
        env_info['mmengine_version'] = "Not installed"

    if torch.cuda.is_available():
        env_info['cuda_version'] = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()
        env_info['cudnn_version'] = f"{cudnn_ver // 1000}.{ (cudnn_ver % 1000) // 100}.{cudnn_ver % 100}" # Format as X.Y.Z
    else:
        env_info['cuda_version'] = "Not available"
        env_info['cudnn_version'] = "Not available"
    
    # Getting GCC version is platform-dependent and might require subprocess,
    # which can be complex. We'll skip it for now or make it optional.
    # For simplicity, we'll report "Not checked"
    env_info['gcc_version'] = "Not checked"

    return env_info

def compare_environments_and_warn(training_env, current_env, log_file_path):
    """Compares training and current environments and prints warnings."""
    if not training_env:
        print("  [Info] Could not retrieve training environment details. Skipping comparison.")
        return

    print("\n--- Environment Mismatch Check ---")
    print(f"  Comparison based on training log: {log_file_path}")
    mismatches = False
    
    # Key fields to compare (and how to compare them - exact or major.minor)
    # (field_name, comparison_type, friendly_name)
    fields_to_compare = [
        ('python_version', 'major.minor', 'Python'),
        ('pytorch_version', 'pytorch_custom', 'PyTorch'), # Changed for custom PyTorch comparison
        ('torchvision_version', 'major.minor', 'TorchVision'),
        ('mmengine_version', 'major.minor', 'MMEngine'),
        ('opencv_version', 'major.minor', 'OpenCV'), 
        ('cuda_version', 'exact', 'CUDA Runtime'), 
        ('cudnn_version', 'major', 'CuDNN'), 
        ('gcc_version', 'skip', 'GCC'),
    ]

    for field, comp_type, name in fields_to_compare:
        train_val = training_env.get(field)
        curr_val = current_env.get(field)

        if field == 'gcc_version' and comp_type == 'skip':
            # print(f"  {name}: Training='{train_val if train_val else 'N/A in log'}', Current='{curr_val}' (Comparison skipped)")
            continue

        if not train_val:
            # print(f"  {name}: Not found in training log. Current version: {curr_val}")
            continue
        
        if curr_val == "Not installed" and train_val and train_val != "Not available":
             print(f"  [Potential Issue] {name}: Training='{train_val}', Current='{curr_val}'")
             mismatches = True
             continue
        elif curr_val == "Not available" and train_val and train_val != "Not available": # Handles cases where CUDA might not be available
            print(f"  [Potential Issue] {name}: Training='{train_val}', Current='{curr_val}'")
            mismatches = True
            continue
        elif not curr_val: # If current value is None or empty after previous checks
            print(f"  [Info] {name}: Training='{train_val}', Current value not available for comparison.")
            continue


        # Attempt to parse versions for typed comparisons
        train_v_parsed = None
        curr_v_parsed = None
        parse_error = False
        try:
            if train_val and train_val not in ["Not available", "Not checked"]:
                train_v_parsed = version.parse(train_val)
            if curr_val and curr_val not in ["Not available", "Not checked", "Not installed"]:
                curr_v_parsed = version.parse(curr_val)
        except version.InvalidVersion:
            parse_error = True # Will fallback to exact string if parsing fails for a typed comparison

        mismatch_found_for_field = False
        
        if comp_type == 'pytorch_custom':
            if not train_v_parsed or not curr_v_parsed or parse_error:
                if train_val != curr_val: # Fallback to exact if parsing failed or not applicable
                    print(f"  [MISMATCH] {name} (parse error/fallback): Training='{train_val}', Current='{curr_val}'")
                    mismatch_found_for_field = True
            elif train_v_parsed.base_version != curr_v_parsed.base_version:
                print(f"  [MISMATCH] {name} (base version): Training='{train_v_parsed.base_version}', Current='{curr_v_parsed.base_version}' (Full: Train='{train_val}', Curr='{curr_val}')")
                mismatch_found_for_field = True
            elif train_v_parsed.local != curr_v_parsed.local:
                # Only report local mismatch if base versions are the same
                # and if at least one has a local version part (e.g. 2.0 vs 2.0+cu118 is a mismatch)
                if train_v_parsed.local or curr_v_parsed.local:
                     print(f"  [MISMATCH] {name} (build metadata): Training='{train_val}', Current='{curr_val}'")
                     mismatch_found_for_field = True
        
        elif comp_type == 'major.minor':
            if not train_v_parsed or not curr_v_parsed or parse_error:
                 if train_val != curr_val: # Fallback
                    print(f"  [MISMATCH] {name} (parse error/fallback): Training='{train_val}', Current='{curr_val}'")
                    mismatch_found_for_field = True
            else:
                train_mm = f"{train_v_parsed.major}.{train_v_parsed.minor}"
                curr_mm = f"{curr_v_parsed.major}.{curr_v_parsed.minor}"
                if train_mm != curr_mm:
                    print(f"  [MISMATCH] {name}: Training='{train_val}' (evaluates to {train_mm}), Current='{curr_val}' (evaluates to {curr_mm})")
                    mismatch_found_for_field = True
        
        elif comp_type == 'major':
            if not train_v_parsed or not curr_v_parsed or parse_error:
                if train_val != curr_val: # Fallback
                    print(f"  [MISMATCH] {name} (parse error/fallback): Training='{train_val}', Current='{curr_val}'")
                    mismatch_found_for_field = True
            else:
                if str(train_v_parsed.major) != str(curr_v_parsed.major):
                    print(f"  [MISMATCH] {name}: Training='{train_val}' (evaluates to major {train_v_parsed.major}), Current='{curr_val}' (evaluates to major {curr_v_parsed.major})")
                    mismatch_found_for_field = True

        elif comp_type == 'exact':
            if train_val != curr_val:
                print(f"  [MISMATCH] {name}: Training='{train_val}', Current='{curr_val}'")
                mismatch_found_for_field = True
        
        if mismatch_found_for_field:
            mismatches = True
        # else:
        #     # Optional: print OK message if needed and values are comparable
        #     if train_val and curr_val and train_val not in ["Not available", "Not checked"] and curr_val not in ["Not available", "Not checked", "Not installed"]:
        #          print(f"  [OK] {name}: Training='{train_val}', Current='{curr_val}'")

    if not mismatches:
        print("  All checked environment components match or are compatible.")
    else:
        print("  [Warning] Environment mismatches detected. This could lead to unexpected behavior or errors.")
    print("--- End of Environment Mismatch Check ---\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MMDetection inference on images')
    # Required arguments
    parser.add_argument('--config', help='Config file path', required=True)
    parser.add_argument('--checkpoint', help='Checkpoint file path (.pth file)', required=True)
    parser.add_argument('--input', help='Image file or directory containing images', required=True)

    # Optional arguments
    parser.add_argument('--output', help='Output directory for visualized results', default=None)
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold for visualization')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')

    # MLflow arguments
    parser.add_argument('--log-mlflow', action='store_true', help='Log model to MLflow')
    parser.add_argument('--mlflow-experiment', default='MMDetection Models', help='MLflow experiment name')
    parser.add_argument('--mlflow-run-name', default=None, help='MLflow run name')

    # Environment check arguments
    parser.add_argument('--training-log-file', default=None, help='Path to the MMDetection training log file for environment comparison.')
    parser.add_argument('--training-work-dir', default=None, help='Path to the training work directory. Used to find the latest log if --training-log-file is not set.')
    parser.add_argument('--check-env-only', action='store_true', help='If set, performs environment checks and PyTorch version compatibility checks then exits without running inference.')
    
    return parser.parse_args()

def process_image(model, visualizer, img_path, output_dir=None, score_thr=0.3):
    """Process a single image and visualize results.
    
    Args:
        model: The MMDetection model
        visualizer: MMDetection visualizer
        img_path: Path to the image
        output_dir: Directory to save visualization results
        score_thr: Score threshold for visualization
        
    Returns:
        Detection results

    """
    # Read image
    img = mmcv.imread(img_path)

    # Run inference
    # In MMDetection 3.x, inference_detector returns DetDataSample or list of DetDataSample
    result = inference_detector(model, img)

    # Visualize results
    if output_dir is not None:
        # Save visualization to file
        out_file = os.path.join(output_dir, os.path.basename(img_path))
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=score_thr,
        )
        print(f"Results saved to {out_file}")
    else:
        # Display visualization directly
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=True,
            wait_time=0,
            pred_score_thr=score_thr,
        )

    return result

def process_batch(model, visualizer, img_paths, output_dir=None, score_thr=0.3):
    """Process a batch of images.
    
    Args:
        model: The MMDetection model
        visualizer: MMDetection visualizer
        img_paths: List of image paths
        output_dir: Directory to save visualization results
        score_thr: Score threshold for visualization
        
    Returns:
        List of detection results

    """
    # Read images
    imgs = [mmcv.imread(img_path) for img_path in img_paths]

    # Run inference in batch mode
    results = inference_detector(model, imgs)

    # Process each result
    for i, (img_path, img, result) in enumerate(zip(img_paths, imgs, results, strict=False)):
        if output_dir is not None:
            # Save visualization to file
            out_file = os.path.join(output_dir, os.path.basename(img_path))
            visualizer.add_datasample(
                f'result_{i}',
                img,
                data_sample=result,
                draw_gt=False,
                show=False,
                wait_time=0,
                out_file=out_file,
                pred_score_thr=score_thr,
            )
            print(f"Results saved to {out_file}")
        else:
            # Display visualization directly (this will show images one after another)
            visualizer.add_datasample(
                f'result_{i}',
                img,
                data_sample=result,
                draw_gt=False,
                show=True,
                wait_time=0 if i == len(imgs) - 1 else 1,  # Wait on last image
                pred_score_thr=score_thr,
            )

    return results

def log_model_to_mlflow(model, config_file, checkpoint_file, experiment_name, run_name=None):
    """Log the model to MLflow.
    
    Args:
        model: The MMDetection model
        config_file: Path to the config file
        checkpoint_file: Path to the checkpoint file
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        
    Returns:
        Model URI

    """
    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Start a run
    with mlflow.start_run(run_name=run_name):
        # Log model parameters from config
        config = Config.fromfile(config_file)

        # Log basic model info
        if hasattr(config.model, 'type'):
            mlflow.log_param("model_type", config.model.type)

        # Extract and log key model parameters
        model_params = {}
        for key, value in config.model.items():
            if isinstance(value, (int, float, str, bool)):
                model_params[f"model.{key}"] = value

        # Log additional parameters like backbone info if available
        if hasattr(config.model, 'backbone') and hasattr(config.model.backbone, 'type'):
            mlflow.log_param("backbone_type", config.model.backbone.type)

        # Log all extracted parameters
        for key, value in model_params.items():
            mlflow.log_param(key, value)

        # Log the config file as an artifact
        mlflow.log_artifact(config_file, "config")

        # Log the checkpoint file as an artifact
        mlflow.log_artifact(checkpoint_file, "checkpoint")

        # Log the PyTorch model
        mlflow.pytorch.log_model(
            model,
            "mmdetection_model",
            # Optionally add code dependencies if available
            code_paths=[os.path.abspath(__file__)],
        )

        # Get the model URI for later reference
        model_uri = mlflow.get_artifact_uri("mmdetection_model")
        print(f"Model logged to MLflow. Model URI: {model_uri}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return model_uri

# CREATE A PATCHED VERSION OF INIT_DETECTOR
def patched_init_detector(config, checkpoint, device='cuda:0'):
    """Initialize a detector from config file with weights_only=False.
    
    This is a patched version of MMDetection's init_detector function to handle 
    the PyTorch 2.6+ security change where torch.load defaults to weights_only=True.
    This patched version temporarily changes torch.load to use weights_only=False
    during checkpoint loading to maintain compatibility with older checkpoints.
    
    Args:
        config (str or Config): Config file path or Config object.
        checkpoint (str): Checkpoint file path.
        device (str): Device to run inference on.
            
    Returns:
        nn.Module: The constructed detector.
    """
    from mmdet.utils import register_all_modules
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint
    from mmengine.config import Config
    import torch
    
    # Register all modules to add them to the registry
    register_all_modules()
    
    # Load config file if it's a string path
    if isinstance(config, str):
        config = Config.fromfile(config)
    
    # Build the model
    model = MODELS.build(config.model)
    
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    if checkpoint is not None:
        # Only apply the fix for PyTorch 2.6+
        pytorch_version = version.parse(torch.__version__.split('+')[0])  # Handle 2.7.0+cu126 format
        needs_weights_only_fix = pytorch_version >= version.parse('2.6.0')
        
        if needs_weights_only_fix:
            # Create a monkey-patched version of torch.load that uses weights_only=False
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
            
            try:
                checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
            finally:
                # Restore the original torch.load function
                torch.load = original_torch_load
        else:
            # For PyTorch < 2.6, use the default behavior (weights_only=False)
            checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    print(f"Starting MMDetection inference script with config: {args.config}")
    
    # --- Environment Mismatch Check ---
    log_file_to_check = args.training_log_file
    if not log_file_to_check and args.training_work_dir:
        if os.path.isdir(args.training_work_dir):
            # Find the latest timestamped subdirectory (e.g., YYYYMMDD_HHMMSS)
            subdirs = []
            for d in os.listdir(args.training_work_dir):
                if os.path.isdir(os.path.join(args.training_work_dir, d)) and re.match(r'^\\d{8}_\\d{6}$', d):
                    subdirs.append(d)
            
            if subdirs:
                subdirs.sort(reverse=True) # Sort to get the latest first
                latest_run_subdir_name = subdirs[0]
                latest_run_dir_path = os.path.join(args.training_work_dir, latest_run_subdir_name)
                
                # Construct the expected log file name based on the subdir name
                expected_log_file_name = f"{latest_run_subdir_name}.log"
                potential_log_file_path = os.path.join(latest_run_dir_path, expected_log_file_name)
                
                if os.path.isfile(potential_log_file_path):
                    log_file_to_check = potential_log_file_path
                    print(f"  [Info] Automatically selected training log: {log_file_to_check}")
                else:
                    # Fallback: if specific log not found, try to find any .log file in the latest run dir
                    log_files = glob.glob(os.path.join(latest_run_dir_path, '*.log'))
                    if log_files:
                        log_files.sort() # Ensure consistent picking
                        log_file_to_check = log_files[0]
                        print(f"  [Info] Automatically selected training log (fallback): {log_file_to_check}")
                    else:
                        print(f"  [Warning] No .log files found in the latest run directory: {latest_run_dir_path}")
            else:
                print(f"  [Warning] No timestamped subdirectories (YYYYMMDD_HHMMSS) found in: {args.training_work_dir}")
        else:
            print(f"  [Warning] Provided training work directory is not a valid directory: {args.training_work_dir}")

    print(f"Log file to check: {log_file_to_check}")
    if log_file_to_check:
        training_env_details = get_training_env_from_log(log_file_to_check)
        print(f"Training environment details: {training_env_details}")
        current_env_details = get_current_env()
        print(f"Current environment details: {current_env_details}")
        compare_environments_and_warn(training_env_details, current_env_details, log_file_to_check)
    elif args.training_log_file or args.training_work_dir: # if user tried to specify but it wasn't found
        print("  [Info] Could not determine training log file for environment comparison.")
    # --- End of Environment Mismatch Check ---
    
    # Display security warning only if using PyTorch 2.6+
    pytorch_version = version.parse(torch.__version__.split('+')[0])
    if pytorch_version >= version.parse('2.6.0'):
        print("\nWARNING: This script uses torch.load with weights_only=False for compatibility with older")
        print("         MMDetection checkpoints. This is less secure when loading models from untrusted sources.")
        print("         Only use this with models from trusted sources.\n")
        print(f"Detected PyTorch {torch.__version__}, applying compatibility fix for checkpoint loading.\n")
    else:
        print(f"Detected PyTorch {torch.__version__}, no compatibility fix needed for checkpoint loading.\n")

    if args.check_env_only:
        print("Environment and compatibility checks complete. Exiting as --check-env-only was specified.")
        sys.exit(0)

    # Build the model from config
    config = Config.fromfile(args.config)

    # For MMDetection 3.x, ensure the visualizer is properly configured
    if hasattr(config, 'visualizer'):
        if not hasattr(config.visualizer, 'vis_backends'):
            config.visualizer.vis_backends = [
                dict(type='LocalVisBackend'),
            ]
    else:
        # Create a default visualizer config if none exists
        config.visualizer = dict(
            type='DetLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend')],
            name='visualizer',
        )

    # Create visualizer
    visualizer = VISUALIZERS.build(config.visualizer)

    # Set dataset meta for visualization
    if hasattr(config, 'dataset_meta'):
        visualizer.dataset_meta = config.dataset_meta
    else:
        # Try to get dataset meta from a common location in config
        for key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if hasattr(config, key) and hasattr(config[key], 'dataset') and hasattr(config[key].dataset, 'metainfo'):
                visualizer.dataset_meta = config[key].dataset.metainfo
                break

    # Initialize the detector
    print(f"Initializing model from {args.config} and {args.checkpoint}")
    model = patched_init_detector(args.config, args.checkpoint, device=args.device)
    print(f"Model initialized on device: {args.device}")

    # Log model to MLflow if requested
    if args.log_mlflow:
        print("Logging model to MLflow...")
        log_model_to_mlflow(
            model,
            args.config,
            args.checkpoint,
            args.mlflow_experiment,
            args.mlflow_run_name,
        )

    # Create output directory if specified
    if args.output:
        mkdir_or_exist(args.output)
        print(f"Output directory: {args.output}")

    start_time = time.time()

    # Process input (single image or directory)
    if os.path.isfile(args.input):
        # Process single image
        print(f"Processing single image: {args.input}")
        process_image(model, visualizer, args.input, args.output, args.score_thr)
    elif os.path.isdir(args.input):
        # Process all images in the directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = []

        # Find all image files in the directory
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.input, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(args.input, f'*{ext.upper()}')))

        if not image_files:
            print(f"No image files found in {args.input}")
            return

        print(f"Found {len(image_files)} images in {args.input}")

        if args.batch_size > 1:
            # Process in batches
            for i in range(0, len(image_files), args.batch_size):
                batch = image_files[i:i + args.batch_size]
                print(f"Processing batch {i//args.batch_size + 1}/{(len(image_files)-1)//args.batch_size + 1} ({len(batch)} images)")
                process_batch(model, visualizer, batch, args.output, args.score_thr)
        else:
            # Process one by one
            for img_path in image_files:
                print(f"Processing {img_path}")
                process_image(model, visualizer, img_path, args.output, args.score_thr)
    else:
        print(f"Error: {args.input} is not a valid file or directory")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
