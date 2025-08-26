# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests>=2.32.3",
#     "tqdm>=4.67.1",
# ]
# ///

"""
Standalone script to download all ONNX CREPE models.
This script uses the new onnxcrepe.download module for consistent model management.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import onnxcrepe
sys.path.insert(0, str(Path(__file__).parent))

try:
    from onnxcrepe.download import download_all_models, get_cache_dir
except ImportError:
    print("Error: Could not import onnxcrepe.download module.")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def main():
    """Download all available CREPE models."""
    print("Downloading all ONNX CREPE models...")
    print(f"Cache directory: {get_cache_dir()}")
    print()
    
    try:
        model_paths = download_all_models(verbose=True)
        
        print(f"\nDownload completed. {len(model_paths)} models available:")
        for model, path in model_paths.items():
            print(f"  {model}: {path}")
        
        if len(model_paths) == 0:
            print("Warning: No models were successfully downloaded.")
            return 1
            
    except Exception as e:
        print(f"Error during download: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
