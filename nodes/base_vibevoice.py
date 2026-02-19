# Created by Fabio Sarracino
# Base class for VibeVoice nodes with common functionality

import logging
import os
import tempfile
import torch
import numpy as np
import re
import gc
import json
from typing import List, Optional, Tuple, Any, Dict

# Setup logging
logger = logging.getLogger("VibeVoice")

# Import for interruption support
try:
    import execution
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False
    logger.warning("Interruption support not available")

# Check for SageAttention availability
try:
    from sageattention import sageattn
    SAGE_AVAILABLE = True
    logger.info("SageAttention available for acceleration")
except ImportError:
    SAGE_AVAILABLE = False
    logger.debug("SageAttention not available - install with: pip install sageattention")

def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_map():
    """Get device map for model loading"""
    device = get_optimal_device()
    # Note: device_map "auto" might work better for MPS in some cases
    return device if device != "mps" else "mps"

# Cache for model scanning to avoid repeated scans
_model_cache = {
    "models": None,
    "last_scan_time": 0,
    "cache_duration": 5,  # Cache for 5 seconds
    "first_load_logged": False  # Track if we've logged the initial scan
}

def get_available_models() -> List[Tuple[str, str]]:
    """Scan models/vibevoice/ directory and return available models

    Returns:
        List of tuples (display_name, folder_path)
    """
    import time

    # Check if we have a valid cache
    current_time = time.time()
    if (_model_cache["models"] is not None and
        current_time - _model_cache["last_scan_time"] < _model_cache["cache_duration"]):
        # Return cached results
        return _model_cache["models"]

    try:
        import folder_paths
        
        # Register vibevoice folder if not present
        if "vibevoice" not in folder_paths.folder_names_and_paths:
            if hasattr(folder_paths, "models_dir"):
                base_path = os.path.join(folder_paths.models_dir, "vibevoice")
            else:
                try:
                    models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                    base_path = os.path.join(os.path.dirname(models_dir), "vibevoice")
                except:
                    # Fallback to a relative path assuming we're in custom_nodes/VibeVoice/nodes
                    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "models", "vibevoice")
            
            folder_paths.add_model_folder_path("vibevoice", base_path)
            
        vibevoice_paths = folder_paths.get_folder_paths("vibevoice")
        
        # Ensure first directory exists if we have paths
        if vibevoice_paths and not os.path.exists(vibevoice_paths[0]):
             try:
                 os.makedirs(vibevoice_paths[0], exist_ok=True)
                 logger.info(f"Created vibevoice models directory: {vibevoice_paths[0]}")
             except:
                 pass

        # First, collect all valid model folders
        valid_folders = []
        seen_folders = set()
        
        for vibevoice_dir in vibevoice_paths:
            if not os.path.exists(vibevoice_dir):
                continue
                
            logger.debug(f"Scanning vibevoice directory: {vibevoice_dir}")
            try:
                folders = os.listdir(vibevoice_dir)
            except:
                continue
                
            for folder in folders:
                if folder in seen_folders:
                    continue
                    
                folder_path = os.path.join(vibevoice_dir, folder)
    
                # Skip hidden folders, loras, and non-directories
                if folder.startswith(".") or folder == "loras" or not os.path.isdir(folder_path):
                    logger.debug(f"Skipping: {folder}")
                    continue
    
                logger.debug(f"Checking folder: {folder}")
                # Check if it's a valid model folder
                if is_valid_model_folder(folder_path):
                    valid_folders.append(folder)
                    seen_folders.add(folder)
                else:
                    logger.debug(f"Folder {folder} is not a valid model folder")

        # Now transform folder names with duplicate detection
        models = []
        for folder in valid_folders:
            display_name = transform_folder_name(folder, valid_folders)
            models.append((folder, display_name))
            logger.debug(f"Found model: {display_name} in folder: {folder}")

        # Sort by display name for consistent ordering
        models.sort(key=lambda x: x[1])

        # Only log on first scan to avoid spam
        if not _model_cache["first_load_logged"]:
            if not models:
                logger.warning("No valid models found in vibevoice directory")
                logger.info(f"Please download models to: {vibevoice_dir}")
            else:
                # Single summary message instead of individual logs
                logger.info(f"Found {len(models)} VibeVoice model(s) available")
            _model_cache["first_load_logged"] = True

        # Cache the results
        _model_cache["models"] = models
        _model_cache["last_scan_time"] = current_time

        return models

    except Exception as e:
        logger.error(f"Error scanning models directory: {e}")
        # Cache empty result on error to avoid repeated failures
        _model_cache["models"] = []
        _model_cache["last_scan_time"] = current_time
        return []

def extract_model_info(folder: str) -> Tuple[str, Optional[str]]:
    """Extract model name and author from folder name

    Args:
        folder: Folder name

    Returns:
        Tuple of (model_name, author_name)

    Examples:
        models--microsoft--VibeVoice-Large -> ('VibeVoice-Large', 'microsoft')
        models--aoi-ot--VibeVoice-Large -> ('VibeVoice-Large', 'aoi-ot')
        VibeVoice-1.5B -> ('VibeVoice-1.5B', None)
    """
    if "--" in folder:
        # HuggingFace cache format: models--author--model
        parts = folder.split("--")
        if len(parts) >= 3:
            author = parts[1]
            model = parts[-1]
            return model, author
        elif len(parts) == 2:
            return parts[-1], None
    return folder, None

def transform_folder_name(folder: str, all_folders: List[str]) -> str:
    """Transform folder name for display, adding author if there are duplicates

    Args:
        folder: Current folder name
        all_folders: List of all folder names to check for duplicates

    Returns:
        Display name with author in parentheses if needed
    """
    model_name, author = extract_model_info(folder)

    # Check if there are other folders with the same model name
    has_duplicate = False
    for other_folder in all_folders:
        if other_folder != folder:
            other_model_name, _ = extract_model_info(other_folder)
            if other_model_name == model_name:
                has_duplicate = True
                break

    # Add author in parentheses if there are duplicates and author is known
    if has_duplicate and author:
        return f"{model_name} ({author})"

    return model_name

def check_folder_has_model_files(folder_path: str) -> bool:
    """Check if a folder directly contains model files (not recursively)

    Args:
        folder_path: Path to check

    Returns:
        True if folder contains config.json and model files
    """
    if not os.path.isdir(folder_path):
        return False

    has_config = os.path.exists(os.path.join(folder_path, "config.json"))
    if not has_config:
        return False

    # Check for various model file formats
    files = os.listdir(folder_path)
    has_model = (
        "pytorch_model.bin" in files or
        "model.safetensors" in files or
        "pytorch_model.bin.index.json" in files or
        "model.safetensors.index.json" in files or
        any(f.startswith("pytorch_model-") and f.endswith(".bin") for f in files) or
        any(f.startswith("model-") and f.endswith(".safetensors") for f in files)
    )

    return has_model

def is_valid_model_folder(folder_path: str, max_depth: int = 4, current_depth: int = 0) -> bool:
    """Recursively check if a folder contains a valid VibeVoice model

    Args:
        folder_path: Path to the folder to check
        max_depth: Maximum recursion depth (default 3)
        current_depth: Current recursion depth

    Returns:
        True if folder or any subfolder contains valid model files
    """
    if current_depth >= max_depth:
        return False

    # Check if current folder has model files
    if check_folder_has_model_files(folder_path):
        return True

    # Recursively check subfolders
    try:
        for item in os.listdir(folder_path):
            # Skip hidden folders and specific folders we want to ignore
            if item.startswith(".") or item in ["loras", "__pycache__"]:
                continue

            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                # Recursively check subfolder
                if is_valid_model_folder(item_path, max_depth, current_depth + 1):
                    return True
    except (PermissionError, OSError):
        pass

    return False

def find_model_files_path_recursive(folder_path: str, max_depth: int = 4, current_depth: int = 0) -> Optional[str]:
    """Recursively find the path containing model files

    Args:
        folder_path: Path to search from
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        Path to the directory containing model files, or None
    """
    if current_depth >= max_depth:
        return None

    # Check if current folder has model files
    if check_folder_has_model_files(folder_path):
        return folder_path

    # Recursively check subfolders
    try:
        for item in os.listdir(folder_path):
            # Skip hidden folders and specific folders we want to ignore
            if item.startswith(".") or item in ["loras", "__pycache__"]:
                continue

            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                # Recursively check subfolder
                result = find_model_files_path_recursive(item_path, max_depth, current_depth + 1)
                if result:
                    return result
    except (PermissionError, OSError):
        pass

    return None

def find_model_files_path(model_folder: str) -> Optional[str]:
    """Find the actual path containing model files

    Args:
        model_folder: Name of the folder in vibevoice directory

    Returns:
        Path to the directory containing model files, or None
    """
    try:
        import folder_paths
        if "vibevoice" not in folder_paths.folder_names_and_paths:
             # Should have been registered by get_available_models, but safe fallback
             if hasattr(folder_paths, "models_dir"):
                base_path = os.path.join(folder_paths.models_dir, "vibevoice")
             else:
                try:
                    models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                    base_path = os.path.join(os.path.dirname(models_dir), "vibevoice")
                except:
                    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "models", "vibevoice")
             folder_paths.add_model_folder_path("vibevoice", base_path)
             
        vibevoice_paths = folder_paths.get_folder_paths("vibevoice")
        
        for vibevoice_dir in vibevoice_paths:
            base_path = os.path.join(vibevoice_dir, model_folder)
            if not os.path.exists(base_path):
                continue
                
            # Use recursive search to find model files
            result = find_model_files_path_recursive(base_path)
    
            if result:
                logger.info(f"Found model files at: {result}")
                return result
                
        logger.warning(f"No valid model files found for: {model_folder}")
        return None

    except Exception as e:
        logger.error(f"Error finding model files: {e}")
        return None

def find_qwen_tokenizer_path(comfyui_models_dir: str) -> Optional[str]:
    """Find Qwen tokenizer using priority system

    Priority:
    1. ComfyUI/models/vibevoice/tokenizer/
    2. ComfyUI/models/vibevoice/models--Qwen--Qwen2.5-1.5B/
    3. HuggingFace cache (if exists)

    Returns:
        Path to tokenizer directory or None
    """
    # Priority 1: Check tokenizer folder
    tokenizer_dir = os.path.join(comfyui_models_dir, "tokenizer")
    if os.path.exists(tokenizer_dir):
        # Check if it contains tokenizer files
        required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
        if all(os.path.exists(os.path.join(tokenizer_dir, f)) for f in required_files):
            logger.info(f"Found Qwen tokenizer in: {tokenizer_dir}")
            return tokenizer_dir

    # Priority 2: Check models--Qwen--Qwen2.5-1.5B folder
    qwen_model_dir = os.path.join(comfyui_models_dir, "models--Qwen--Qwen2.5-1.5B")
    if os.path.exists(qwen_model_dir):
        # Check snapshots folder
        snapshots_dir = os.path.join(qwen_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            for snapshot in os.listdir(snapshots_dir):
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if os.path.isdir(snapshot_path):
                    # Check if it contains tokenizer files
                    if os.path.exists(os.path.join(snapshot_path, "tokenizer_config.json")):
                        logger.info(f"Found Qwen tokenizer in model cache: {snapshot_path}")
                        return snapshot_path

    # Priority 3: Check HuggingFace cache
    hf_cache_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
    ]

    for cache_path in hf_cache_paths:
        if cache_path and os.path.exists(cache_path):
            qwen_cache = os.path.join(cache_path, "models--Qwen--Qwen2.5-1.5B")
            if os.path.exists(qwen_cache):
                snapshots_dir = os.path.join(qwen_cache, "snapshots")
                if os.path.exists(snapshots_dir):
                    for snapshot in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        if os.path.isdir(snapshot_path):
                            if os.path.exists(os.path.join(snapshot_path, "tokenizer_config.json")):
                                logger.info(f"Found Qwen tokenizer in HF cache: {snapshot_path}")
                                return snapshot_path

    return None

def detect_model_quantization(model_path: str) -> Optional[str]:
    """Detect if model is quantized from config files

    Args:
        model_path: Path to the model directory

    Returns:
        '4bit', '8bit', or None
    """
    try:
        # Check for quantization_config.json first
        quant_config_path = os.path.join(model_path, "quantization_config.json")
        if os.path.exists(quant_config_path):
            with open(quant_config_path, 'r') as f:
                quant_config = json.load(f)
                if quant_config.get("load_in_4bit"):
                    return "4bit"
                if quant_config.get("load_in_8bit"):
                    return "8bit"

        # Check main config.json
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "quantization_config" in config:
                    if config["quantization_config"].get("load_in_4bit"):
                        return "4bit"
                    if config["quantization_config"].get("load_in_8bit"):
                        return "8bit"
                    if config["quantization_config"].get("bits") == 4:
                        return "4bit"

    except Exception as e:
        logger.debug(f"Could not detect quantization: {e}")

    return None

class BaseVibeVoiceNode:
    """Base class for VibeVoice nodes containing common functionality"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_folder = None
        self.current_attention_type = None
        self.current_quantize_llm = "full precision"
        self.current_lora_path = None
        # LoRA component flags (overridable by node instances)
        self.use_llm_lora = True
        self.use_diffusion_head_lora = True
        self.use_acoustic_connector_lora = True
        self.use_semantic_connector_lora = True
    
    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.current_model_folder = None
            self.current_quantize_llm = "full precision"
            
            # Force garbage collection and clear CUDA cache if available
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import sys
            import os
            
            # Add vvembed to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            vvembed_path = os.path.join(parent_dir, 'vvembed')
            
            if vvembed_path not in sys.path:
                sys.path.insert(0, vvembed_path)
            
            # Import from embedded version
            from modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            
            logger.info(f"Using embedded VibeVoice from {vvembed_path}")
            return None, VibeVoiceForConditionalGenerationInference
            
        except ImportError as e:
            logger.error(f"Embedded VibeVoice import failed: {e}")
            
            # Try fallback to installed version if available
            try:
                import vibevoice
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                logger.warning("Falling back to system-installed VibeVoice")
                return vibevoice, VibeVoiceForConditionalGenerationInference
            except ImportError:
                pass
            
            raise Exception(
                "VibeVoice embedded module import failed. Please ensure the vvembed folder exists "
                "and transformers>=4.51.3 is installed."
            )
    
    def _apply_lora(self, lora_path: str):
        """Apply LoRA adapters to the model"""
        try:
            logger.info(f"Starting LoRA application from path: {lora_path}")

            # Check component flags
            use_llm = getattr(self, 'use_llm_lora', True)
            use_diffusion = getattr(self, 'use_diffusion_head_lora', True)
            use_acoustic = getattr(self, 'use_acoustic_connector_lora', True)
            use_semantic = getattr(self, 'use_semantic_connector_lora', True)

            logger.info(f"LoRA component flags - LLM: {use_llm}, Diffusion: {use_diffusion}, Acoustic: {use_acoustic}, Semantic: {use_semantic}")

            if not any([use_llm, use_diffusion, use_acoustic, use_semantic]):
                logger.info("All LoRA components disabled, skipping LoRA application")
                return

            # Apply LLM LoRA adapter if requested
            if use_llm:
                # Check if adapter files exist
                adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
                adapter_bin_path = os.path.join(lora_path, "adapter_model.bin")
                adapter_config = os.path.join(lora_path, "adapter_config.json")

                has_adapter = os.path.exists(adapter_model_path) or os.path.exists(adapter_bin_path)

                if has_adapter and os.path.exists(adapter_config):
                    try:
                        from peft import PeftModel
                        base_lm = getattr(self.model.model, 'language_model', None)
                        if base_lm is not None:
                            logger.info(f"Applying LLM LoRA adapter from: {lora_path}")
                            lora_wrapped = PeftModel.from_pretrained(base_lm, lora_path, is_trainable=False)
                            device = next(self.model.parameters()).device
                            dtype = next(self.model.parameters()).dtype
                            lora_wrapped = lora_wrapped.to(device=device, dtype=dtype)
                            self.model.model.language_model = lora_wrapped
                            logger.info("LLM LoRA adapter successfully applied")
                    except ImportError:
                        logger.warning("PEFT library not available for LLM LoRA")
                    except Exception as e:
                        logger.warning(f"Failed to apply LLM LoRA: {e}")
                else:
                    logger.info(f"No LLM LoRA adapter files found in {lora_path}, skipping LLM LoRA")

            # Helper function to load state dict into module
            def _load_state_dict_into(module, folder):
                if module is None:
                    logger.warning(f"Module is None, cannot load state dict from {folder}")
                    return False
                if not os.path.isdir(folder):
                    logger.warning(f"Folder does not exist: {folder}")
                    return False

                try:
                    # Try safetensors first
                    safetensor_path = os.path.join(folder, "model.safetensors")
                    if os.path.exists(safetensor_path):
                        try:
                            import safetensors.torch as st
                            logger.info(f"Loading safetensor from: {safetensor_path}")
                            state_dict = st.load_file(safetensor_path)
                            logger.info(f"Loaded state dict with {len(state_dict)} keys")

                            # Get device and dtype from module
                            device = next(module.parameters()).device
                            dtype = next(module.parameters()).dtype
                            logger.info(f"Module device: {device}, dtype: {dtype}")

                            # Convert state dict to correct device and dtype
                            for key in state_dict:
                                state_dict[key] = state_dict[key].to(device=device, dtype=dtype)

                            # Load with strict=False to allow partial loading
                            missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

                            if missing_keys:
                                logger.warning(f"Missing keys when loading state dict ({len(missing_keys)} total): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                            if unexpected_keys:
                                logger.warning(f"Unexpected keys when loading state dict ({len(unexpected_keys)} total): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")

                            # Log success even with missing keys if most were loaded
                            total_keys = len(state_dict)
                            if missing_keys:
                                logger.info(f"Loaded {total_keys} keys from LoRA, {len(missing_keys)} keys not found in model")

                            logger.info("Successfully loaded state dict into module")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to load safetensors: {e}")
                            import traceback
                            logger.debug(f"Traceback: {traceback.format_exc()}")

                    # Fallback to PyTorch format
                    pytorch_path = os.path.join(folder, "pytorch_model.bin")
                    if os.path.exists(pytorch_path):
                        logger.info(f"Loading pytorch model from: {pytorch_path}")
                        state_dict = torch.load(pytorch_path, map_location="cpu")
                        logger.info(f"Loaded state dict with {len(state_dict)} keys")

                        # Get device and dtype from module
                        device = next(module.parameters()).device
                        dtype = next(module.parameters()).dtype

                        # Convert state dict to correct device and dtype
                        for key in state_dict:
                            state_dict[key] = state_dict[key].to(device=device, dtype=dtype)

                        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

                        if missing_keys:
                            logger.warning(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                        if unexpected_keys:
                            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")

                        logger.info("Successfully loaded pytorch model into module")
                        return True
                    else:
                        logger.warning(f"No model file found in {folder}")
                        logger.warning(f"Looked for: {safetensor_path} and {pytorch_path}")

                except Exception as e:
                    logger.error(f"Failed to load state dict from {folder}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                return False

            # Load diffusion head if requested
            if use_diffusion:
                diffusion_path = os.path.join(lora_path, "diffusion_head")
                if os.path.exists(diffusion_path):
                    logger.info(f"Found diffusion_head directory at: {diffusion_path}")

                    # The diffusion head is called 'prediction_head' in VibeVoice
                    module = getattr(self.model.model, 'prediction_head', None)
                    if module:
                        logger.info("Found prediction_head module in model")

                        # Check model compatibility by looking at dimensions
                        skip_loading = False
                        try:
                            # Get hidden size from the module
                            if hasattr(module, 'cond_proj') and hasattr(module.cond_proj, 'weight'):
                                model_hidden_size = module.cond_proj.weight.shape[0]
                                logger.info(f"Current model prediction_head hidden size: {model_hidden_size}")

                                # Check LoRA dimensions
                                safetensor_path = os.path.join(diffusion_path, "model.safetensors")
                                if os.path.exists(safetensor_path):
                                    import safetensors.torch as st
                                    lora_state = st.load_file(safetensor_path)
                                    if 'cond_proj.weight' in lora_state:
                                        lora_hidden_size = lora_state['cond_proj.weight'].shape[0]
                                        logger.info(f"LoRA diffusion head hidden size: {lora_hidden_size}")

                                        if model_hidden_size != lora_hidden_size:
                                            skip_loading = True
                                            if lora_hidden_size == 3584:
                                                logger.error("="*60)
                                                logger.error("LoRA MODEL MISMATCH!")
                                                logger.error(f"This LoRA was trained on VibeVoice-Large (hidden_size=3584)")
                                                if model_hidden_size == 1536:
                                                    logger.error(f"You are using VibeVoice-1.5B (hidden_size=1536)")
                                                    logger.error("Please switch to 'VibeVoice-Large' model to use this LoRA")
                                                else:
                                                    logger.error(f"Your model has hidden_size={model_hidden_size}")
                                                    logger.error("Please use VibeVoice-Large (non-quantized) model")
                                                logger.error("="*60)
                                                logger.error("Skipping LoRA loading due to incompatible model")
                                            elif lora_hidden_size == 1536:
                                                logger.error("="*60)
                                                logger.error("LoRA MODEL MISMATCH!")
                                                logger.error(f"This LoRA was trained on VibeVoice-1.5B (hidden_size=1536)")
                                                logger.error(f"You are using a model with hidden_size={model_hidden_size}")
                                                logger.error("Please switch to 'VibeVoice-1.5B' model to use this LoRA")
                                                logger.error("="*60)
                                                logger.error("Skipping LoRA loading due to incompatible model")
                        except Exception as e:
                            logger.debug(f"Could not check model compatibility: {e}")

                        # Only attempt to load if compatible
                        if not skip_loading:
                            if _load_state_dict_into(module, diffusion_path):
                                logger.info("Diffusion head LoRA loaded successfully into prediction_head")
                            else:
                                logger.warning("Failed to load diffusion head LoRA")
                        else:
                            logger.info("Diffusion head LoRA loading skipped due to model mismatch")
                    else:
                        logger.warning("Model does not have prediction_head attribute")
                        # Debug: list available attributes
                        attrs = [a for a in dir(self.model.model) if not a.startswith('_')]
                        logger.debug(f"Available model.model attributes: {attrs[:15]}...")
                else:
                    logger.info(f"No diffusion_head directory found at: {diffusion_path}")

            # Load acoustic connector if requested
            if use_acoustic:
                acoustic_path = os.path.join(lora_path, "acoustic_connector")
                if os.path.exists(acoustic_path):
                    module = getattr(self.model.model, 'acoustic_connector', None)
                    if module and _load_state_dict_into(module, acoustic_path):
                        logger.info("Acoustic connector LoRA loaded")

            # Load semantic connector if requested
            if use_semantic:
                semantic_path = os.path.join(lora_path, "semantic_connector")
                if os.path.exists(semantic_path):
                    module = getattr(self.model.model, 'semantic_connector', None)
                    if module and _load_state_dict_into(module, semantic_path):
                        logger.info("Semantic connector LoRA loaded")


            # Log summary of what was loaded
            logger.info("LoRA application completed")

        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't fail the entire load, just log the error

    def _verify_quantization(self, expected_mode: str):
        """Verify that quantization was actually applied correctly"""
        try:
            quantized_layers = []
            fp_layers = []

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module_type = type(module).__name__

                    if 'Linear8bitLt' in module_type or '8bit' in module_type.lower():
                        quantized_layers.append((name, '8bit'))
                    elif 'Linear4bit' in module_type or '4bit' in module_type.lower():
                        quantized_layers.append((name, '4bit'))
                    else:
                        fp_layers.append(name)

            # Concise summary
            total_linear = len(quantized_layers) + len(fp_layers)

            if len(quantized_layers) > 0:
                pct = 100 * len(quantized_layers) / total_linear
                logger.info(f"✅ {expected_mode} quantization: {len(quantized_layers)}/{total_linear} layers ({pct:.1f}%)")
            else:
                logger.warning(f"⚠️ No {expected_mode} quantization detected")

        except Exception as e:
            logger.debug(f"Could not verify quantization: {e}")

    def _apply_sage_attention(self):
        """Apply SageAttention to the loaded model by monkey-patching attention layers"""
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            
            # Counter for patched layers
            patched_count = 0
            
            def patch_attention_forward(module):
                """Recursively patch attention layers to use SageAttention"""
                nonlocal patched_count
                
                # Check if this module has scaled_dot_product_attention
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    # Create wrapper that replaces F.scaled_dot_product_attention with sageattn
                    def sage_forward(*args, **kwargs):
                        # Temporarily replace F.scaled_dot_product_attention
                        original_sdpa = F.scaled_dot_product_attention
                        
                        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
                            """Wrapper that converts sdpa calls to sageattn"""
                            # Log any unexpected parameters for debugging
                            if kwargs:
                                unexpected_params = list(kwargs.keys())
                                logger.debug(f"SageAttention: Ignoring unsupported parameters: {unexpected_params}")
                            
                            try:
                                # SageAttention expects tensors in specific format
                                # Transformers typically use (batch, heads, seq_len, head_dim)
                                
                                # Check tensor dimensions to determine layout
                                if query.dim() == 4:
                                    # 4D tensor: (batch, heads, seq, dim)
                                    batch_size = query.shape[0]
                                    num_heads = query.shape[1]
                                    seq_len_q = query.shape[2]
                                    seq_len_k = key.shape[2]
                                    head_dim = query.shape[3]
                                    
                                    # Reshape to (batch*heads, seq, dim) for HND layout
                                    query_reshaped = query.reshape(batch_size * num_heads, seq_len_q, head_dim)
                                    key_reshaped = key.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    value_reshaped = value.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    
                                    # Call sageattn with HND layout
                                    output = sageattn(
                                        query_reshaped, key_reshaped, value_reshaped,
                                        is_causal=is_causal,
                                        tensor_layout="HND"  # Heads*batch, seqN, Dim
                                    )
                                    
                                    # Output should be (batch*heads, seq_len_q, head_dim)
                                    # Reshape back to (batch, heads, seq, dim)
                                    if output.dim() == 3:
                                        output = output.reshape(batch_size, num_heads, seq_len_q, head_dim)
                                    
                                    return output
                                else:
                                    # For 3D tensors, assume they're already in HND format
                                    output = sageattn(
                                        query, key, value,
                                        is_causal=is_causal,
                                        tensor_layout="HND"
                                    )
                                    return output
                                    
                            except Exception as e:
                                # If SageAttention fails, fall back to original implementation
                                logger.debug(f"SageAttention failed, using original: {e}")
                                # Call with proper arguments - scale is a keyword argument in PyTorch 2.0+
                                # Pass through any additional kwargs that the original sdpa might support
                                if scale is not None:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
                                else:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                        
                        # Replace the function
                        F.scaled_dot_product_attention = sage_sdpa
                        
                        try:
                            # Call original forward with patched attention
                            result = original_forward(*args, **kwargs)
                        finally:
                            # Restore original function
                            F.scaled_dot_product_attention = original_sdpa
                        
                        return result
                    
                    # Check if this module likely uses attention
                    # Look for common attention module names
                    module_name = module.__class__.__name__.lower()
                    if any(name in module_name for name in ['attention', 'attn', 'multihead']):
                        module.forward = sage_forward
                        patched_count += 1
                
                # Recursively patch child modules
                for child in module.children():
                    patch_attention_forward(child)
            
            # Apply patching to the entire model
            patch_attention_forward(self.model)
            
            logger.info(f"Patched {patched_count} attention layers with SageAttention")
            
            if patched_count == 0:
                logger.warning("No attention layers found to patch - SageAttention may not be applied")
                
        except Exception as e:
            logger.error(f"Failed to apply SageAttention: {e}")
            logger.warning("Continuing with standard attention implementation")
    
    def load_model(self, model_name: str, model_folder: str, attention_type: str = "auto", quantize_llm: str = "full precision", lora_path: str = None):
        """Load VibeVoice model with specified attention implementation and optional LoRA

        Args:
            model_name: The display name of the model (e.g., "VibeVoice-Large")
            model_folder: The folder name in models/vibevoice/ containing the model
            attention_type: The attention implementation to use
            quantize_llm: LLM quantization mode ("full precision", "8bit", or "4bit")
            lora_path: Optional path to LoRA adapter directory
        """
        # Check if we need to reload model due to attention type, quantization, or LoRA change
        current_attention = getattr(self, 'current_attention_type', None)
        current_quantize_llm = getattr(self, 'current_quantize_llm', 'full precision')
        current_lora = getattr(self, 'current_lora_path', None)
        lora_changed = (current_lora or "") != (lora_path or "")
        quantize_changed = current_quantize_llm != quantize_llm

        if (self.model is None or
            getattr(self, 'current_model_folder', None) != model_folder or
            current_attention != attention_type or
            quantize_changed or
            lora_changed):
            
            # Free existing model before loading new one (important for attention type, quantization, or LoRA changes)
            if self.model is not None and (current_attention != attention_type or quantize_changed or getattr(self, 'current_model_folder', None) != model_folder or lora_changed):
                logger.info(f"Freeing existing model before loading with new settings (attention: {current_attention} -> {attention_type}, quantize: {current_quantize_llm} -> {quantize_llm}, LoRA: {current_lora} -> {lora_path})")
                self.free_memory()
            
            try:
                vibevoice, VibeVoiceInferenceModel = self._check_dependencies()
                
                # Set ComfyUI models directory
                import folder_paths
                models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                comfyui_models_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
                os.makedirs(comfyui_models_dir, exist_ok=True)
                
                # Import time for timing
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Get the actual model path using our discovery function
                model_full_path = os.path.join(comfyui_models_dir, model_folder)

                # Find where the actual model files are
                model_files_path = find_model_files_path(model_folder)

                if not model_files_path:
                    raise Exception(f"No valid model files found in {model_full_path}. Please ensure the model is properly downloaded.")

                logger.info(f"Found model files at: {model_files_path}")

                # Check if model files are in a 4bit subfolder
                use_4bit_subfolder = False
                actual_model_path = model_files_path
                if model_files_path.endswith(os.sep + "4bit"):
                    # If files are in 4bit subfolder, use parent path and set subfolder
                    actual_model_path = os.path.dirname(model_files_path)
                    use_4bit_subfolder = True
                    logger.info(f"Model uses 4bit subfolder structure")

                # Detect if model is quantized
                quantization = detect_model_quantization(model_files_path)
                if quantization:
                    logger.info(f"Detected {quantization} quantization")
                
                # Check if this is a quantized model
                is_quantized_4bit = quantization == "4bit"
                is_quantized_8bit = quantization == "8bit"
                is_quantized = is_quantized_4bit or is_quantized_8bit
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "cache_dir": comfyui_models_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": get_device_map(),
                }
                
                # Handle quantized model loading
                if is_quantized_4bit or is_quantized_8bit:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("Quantized models require a CUDA GPU. Please use standard models on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        if is_quantized_4bit:
                            logger.info("Loading 4-bit quantized model with bitsandbytes...")
                            # Configure 4-bit quantization
                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type='nf4'
                            )
                            if use_4bit_subfolder:
                                model_kwargs["subfolder"] = "4bit"
                                logger.info("Using subfolder='4bit' for loading")
                        else:  # 8-bit
                            logger.info("Loading 8-bit quantized model with bitsandbytes...")
                            # Configure 8-bit quantization
                            bnb_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                bnb_8bit_compute_dtype=torch.bfloat16
                            )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "cuda"  # Force CUDA for quantized models

                    except ImportError:
                        raise Exception(
                            "Quantized models require 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use the standard VibeVoice models instead."
                        )

                # Handle LLM-only 8-bit quantization (for non-quantized models) - EXPERIMENTAL
                elif quantize_llm == "8bit" and not is_quantized:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU. Please use 'full precision' on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        logger.info("Quantizing LLM component to 8-bit...")
                        # Configure 8-bit quantization for LLM only
                        # CRITICAL: Must skip all audio-related components to prevent noise
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.bfloat16,
                            # Skip ALL audio-critical components (same as 4bit + more conservative)
                            llm_int8_skip_modules=[
                                "lm_head",              # Output projection
                                "prediction_head",      # Diffusion head - CRITICAL for audio quality
                                "acoustic_connector",   # Audio->LLM projection - CRITICAL
                                "semantic_connector",   # Semantic->LLM projection - CRITICAL
                                "acoustic_tokenizer",   # VAE encoder/decoder for audio
                                "semantic_tokenizer",   # VAE encoder for semantics
                            ],
                            # Ultra-conservative outlier threshold (lower = more fp16 processing)
                            # Default is 6.0, but audio/diffusion models need 3.0-4.0 for stability
                            llm_int8_threshold=3.0,
                            # Disable fp16 weights (use int8 storage)
                            llm_int8_has_fp16_weight=False,
                        )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "auto"

                        # Flag for post-load verification
                        model_kwargs["_quantization_mode"] = "8bit"

                    except ImportError:
                        raise Exception(
                            "LLM quantization requires 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use 'full precision' mode instead."
                        )

                # Handle LLM-only 4-bit quantization (for non-quantized models)
                elif quantize_llm == "4bit" and not is_quantized:
                    # Check if CUDA is available (required for quantization)
                    if not torch.cuda.is_available():
                        raise Exception("LLM quantization requires a CUDA GPU. Please use 'full precision' on CPU/MPS.")

                    # Try to import bitsandbytes
                    try:
                        from transformers import BitsAndBytesConfig

                        logger.info("Quantizing LLM component to 4-bit...")
                        # Configure 4-bit quantization for LLM only
                        # Note: lm_head must be skipped to avoid bitsandbytes assertion errors
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                            # Skip lm_head and non-LLM components to avoid errors
                            llm_int8_skip_modules=["lm_head", "prediction_head", "acoustic_connector", "semantic_connector", "diffusion_head"]
                        )

                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "auto"
                        logger.info("LLM will be quantized to 4-bit, diffusion head and connectors remain at full precision")

                        # Flag for post-load verification
                        model_kwargs["_quantization_mode"] = "4bit"

                    except ImportError:
                        raise Exception(
                            "LLM quantization requires 'bitsandbytes' library.\n"
                            "Please install it with: pip install bitsandbytes\n"
                            "Or use 'full precision' mode instead."
                        )

                # Set attention implementation based on user selection
                use_sage_attention = False
                if attention_type == "sage":
                    # SageAttention requires special handling - can't be set via attn_implementation
                    if not SAGE_AVAILABLE:
                        logger.warning("SageAttention not installed, falling back to sdpa")
                        logger.warning("Install with: pip install sageattention")
                        model_kwargs["attn_implementation"] = "sdpa"
                    elif not torch.cuda.is_available():
                        logger.warning("SageAttention requires CUDA GPU, falling back to sdpa")
                        model_kwargs["attn_implementation"] = "sdpa"
                    else:
                        # Don't set attn_implementation for sage, will apply after loading
                        use_sage_attention = True
                        logger.info("Will apply SageAttention after model loading")
                elif attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    # Auto mode - let transformers decide the best implementation
                    logger.info("Using auto attention implementation selection")
                
                # Load the model from local path only
                model_kwargs["local_files_only"] = True

                # Extract quantization mode flag before loading (it's not a model parameter)
                quant_mode = model_kwargs.pop("_quantization_mode", None)

                try:
                    # Use the correct path (parent if 4bit subfolder is used)
                    logger.info(f"Loading model from: {actual_model_path}")
                    if is_quantized:
                        logger.info(f"Loading {quantization} quantized model...")
                        if use_4bit_subfolder:
                            logger.info(f"Using parent path with subfolder='4bit'")

                    self.model = VibeVoiceInferenceModel.from_pretrained(
                        actual_model_path,
                        **model_kwargs
                    )
                except Exception as e:
                    logger.error(f"Failed to load model from {model_files_path}: {e}")
                    raise Exception(
                        f"Failed to load model from {model_files_path}.\n"
                        f"Please ensure the model files are complete and properly downloaded.\n"
                        f"Required files: config.json, pytorch_model.bin or model safetensors\n"
                        f"Error: {str(e)}"
                    )

                elapsed = time.time() - start_time
                logger.info(f"Model loaded in {elapsed:.2f} seconds")

                # Verify quantization if requested (quant_mode was extracted earlier)
                if quant_mode:
                    self._verify_quantization(quant_mode)

                # Verify model was loaded
                if self.model is None:
                    raise Exception("Model failed to load - model is None after loading")

                # Load processor with proper error handling
                from processor.vibevoice_processor import VibeVoiceProcessor

                logger.info("Loading VibeVoice processor...")
                processor_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": comfyui_models_dir,
                    "local_files_only": True
                }

                # Add subfolder if needed
                if use_4bit_subfolder:
                    processor_kwargs["subfolder"] = "4bit"

                # Pre-check for Qwen tokenizer - REQUIRED
                tokenizer_path = find_qwen_tokenizer_path(comfyui_models_dir)
                if not tokenizer_path:
                    # Tokenizer is required - fail early with clear instructions
                    logger.error("="*60)
                    logger.error("QWEN TOKENIZER NOT FOUND!")
                    logger.error("The VibeVoice processor requires the Qwen2.5-1.5B tokenizer.")
                    logger.error("")
                    logger.error("To fix this, please download the tokenizer:")
                    logger.error("1. Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                    logger.error("   Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json")
                    logger.error("2. Place files in ONE of these locations (in order of priority):")
                    logger.error(f"   - {os.path.join(comfyui_models_dir, 'tokenizer')}/ (RECOMMENDED)")
                    logger.error(f"   - {os.path.join(comfyui_models_dir, 'models--Qwen--Qwen2.5-1.5B')}/snapshots/[hash]/")
                    logger.error("3. Restart ComfyUI and try again")
                    logger.error("="*60)
                    raise Exception(
                        "Qwen tokenizer not found. Please download it manually.\n"
                        "Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                        "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                        f"Place files in: {os.path.join(comfyui_models_dir, 'tokenizer')}/"
                    )

                # Validate that all required tokenizer files exist
                required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
                missing_files = []
                for file_name in required_files:
                    file_path = os.path.join(tokenizer_path, file_name)
                    if not os.path.exists(file_path):
                        missing_files.append(file_name)

                if missing_files:
                    logger.error("="*60)
                    logger.error(f"TOKENIZER IS INCOMPLETE!")
                    logger.error(f"Tokenizer folder found at: {tokenizer_path}")
                    logger.error(f"But missing required files: {', '.join(missing_files)}")
                    logger.error("")
                    logger.error("Please download ALL required files from:")
                    logger.error("https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                    logger.error("Required files:")
                    logger.error("  - tokenizer_config.json")
                    logger.error("  - vocab.json")
                    logger.error("  - merges.txt")
                    logger.error("  - tokenizer.json (optional but recommended)")
                    logger.error("="*60)
                    raise Exception(
                        f"Tokenizer is incomplete. Missing files: {', '.join(missing_files)}\n"
                        "Please download ALL required files from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                        "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                        f"Place them in: {tokenizer_path}/"
                    )

                logger.info(f"Found complete tokenizer at: {tokenizer_path}")
                # Override the language model path to use local tokenizer
                processor_kwargs["language_model_pretrained_name"] = tokenizer_path
                # Remove cache_dir to avoid HuggingFace cache interference
                processor_kwargs.pop('cache_dir', None)

                try:
                    # Load processor from same path as model
                    self.processor = VibeVoiceProcessor.from_pretrained(
                        actual_model_path,
                        **processor_kwargs
                    )
                except Exception as proc_error:
                    logger.warning(f"Failed to load processor from {model_files_path}: {proc_error}")

                    # Check if error is about missing Qwen tokenizer
                    if ("Qwen" in str(proc_error) or "tokenizer" in str(proc_error).lower()):
                        logger.info("Processor needs Qwen tokenizer. Searching for tokenizer...")

                        # Try to find tokenizer using priority system
                        tokenizer_path = find_qwen_tokenizer_path(comfyui_models_dir)

                        if tokenizer_path:
                            logger.info(f"Found tokenizer at: {tokenizer_path}")
                            # Try to load processor with tokenizer path hint
                            try:
                                from transformers import AutoTokenizer
                                # Load tokenizer from the found path
                                tokenizer = AutoTokenizer.from_pretrained(
                                    tokenizer_path,
                                    trust_remote_code=True,
                                    local_files_only=True
                                )
                                logger.info("Qwen tokenizer loaded successfully from local path")
                                # Store for later use if needed
                                self._temp_tokenizer = tokenizer
                            except Exception as tok_error:
                                logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {tok_error}")
                        else:
                            logger.error("="*60)
                            logger.error("QWEN TOKENIZER NOT FOUND!")
                            logger.error("The VibeVoice processor requires the Qwen2.5-1.5B tokenizer.")
                            logger.error("")
                            logger.error("To fix this, please download the tokenizer:")
                            logger.error("1. Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
                            logger.error("   Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json")
                            logger.error("2. Place files in ONE of these locations:")
                            logger.error(f"   - {os.path.join(comfyui_models_dir, 'tokenizer')}/")
                            logger.error(f"   - {os.path.join(comfyui_models_dir, 'models--Qwen--Qwen2.5-1.5B')}/snapshots/[hash]/")
                            logger.error("3. Restart ComfyUI and try again")
                            logger.error("="*60)
                            raise Exception(
                                "Qwen tokenizer not found. Please download it manually.\n"
                                "Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                                "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                                f"Place tokenizer files in: {os.path.join(comfyui_models_dir, 'tokenizer')}/"
                            )
                    
                    logger.info("Attempting to load processor with fallback method...")
                    
                    # Fallback: try loading without subfolder
                    try:
                        if "subfolder" in processor_kwargs:
                            del processor_kwargs["subfolder"]
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_files_path,
                            **processor_kwargs
                        )
                    except Exception as fallback_error:
                        logger.error(f"Processor loading failed completely: {fallback_error}")
                        # Check if it's still about Qwen tokenizer
                        if "Qwen" in str(fallback_error):
                            tokenizer_path = find_qwen_tokenizer_path(comfyui_models_dir)
                            if not tokenizer_path:
                                raise Exception(
                                    f"Failed to load VibeVoice processor: Missing Qwen tokenizer.\n"
                                    f"Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                                    f"Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                                    f"Place files in: {os.path.join(comfyui_models_dir, 'tokenizer')}/"
                                )

                        raise Exception(
                            f"Failed to load VibeVoice processor. Error: {fallback_error}\n"
                            f"Please ensure transformers>=4.51.3 is installed."
                        )
                
                # Move to appropriate device (skip for quantized models as they use device_map)
                # Skip device movement for both pre-quantized models and LLM-quantized models
                is_llm_quantized = quantize_llm != "full precision"
                if not is_quantized and not is_llm_quantized:
                    device = get_optimal_device()
                    if device == "cuda":
                        self.model = self.model.cuda()
                    elif device == "mps":
                        self.model = self.model.to("mps")
                else:
                    logger.info("Quantized model already mapped to device via device_map")
                
                # Apply SageAttention if requested and available
                if use_sage_attention and SAGE_AVAILABLE:
                    self._apply_sage_attention()
                    logger.info("SageAttention successfully applied to model")

                # Apply LoRA if provided and path exists
                if lora_path and os.path.isdir(lora_path):
                    self._apply_lora(lora_path)

                self.current_model_folder = model_folder
                self.current_attention_type = attention_type
                self.current_quantize_llm = quantize_llm
                self.current_lora_path = lora_path
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)

    def _adjust_voice_speed(self, audio_np: np.ndarray, speed_factor: float, sample_rate: int = 24000) -> np.ndarray:
        """Adjust voice speed using time-stretching without changing pitch significantly

        Args:
            audio_np: Input audio array
            speed_factor: Speed adjustment (0.75 = 25% slower, 1.25 = 25% faster)
            sample_rate: Sample rate of the audio

        Returns:
            Speed-adjusted audio array
        """
        if speed_factor == 1.0:
            return audio_np  # No change needed

        # Calculate new length
        original_length = len(audio_np)
        target_length = int(original_length / speed_factor)

        # Use linear interpolation for time-stretching
        # This is a simple approach that works reasonably well for small speed changes
        original_indices = np.arange(original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)

        # Interpolate the audio to the new length
        adjusted_audio = np.interp(target_indices, original_indices, audio_np)

        logger.info(f"Adjusted voice speed by factor {speed_factor:.2f} ({original_length} -> {target_length} samples)")

        return adjusted_audio.astype(np.float32)

    def _prepare_audio_from_comfyui(self, voice_audio, target_sample_rate: int = 24000, speed_factor: float = 1.0) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format to numpy array"""
        if voice_audio is None:
            return None
            
        # Extract waveform from ComfyUI audio format
        if isinstance(voice_audio, dict) and "waveform" in voice_audio:
            waveform = voice_audio["waveform"]
            input_sample_rate = voice_audio.get("sample_rate", target_sample_rate)
            
            # Convert to numpy (handling BFloat16 tensors)
            if isinstance(waveform, torch.Tensor):
                # Convert to float32 first as numpy doesn't support BFloat16
                audio_np = waveform.cpu().float().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different audio shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0, 0, :]  # Take first batch, first channel
            elif audio_np.ndim == 2:  # (channels, samples)
                audio_np = audio_np[0, :]  # Take first channel
            # If 1D, leave as is
            
            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length), 
                                   np.arange(len(audio_np)), audio_np)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)  # Normalize

            # Apply speed adjustment if requested
            if speed_factor != 1.0:
                audio_np = self._adjust_voice_speed(audio_np, speed_factor, target_sample_rate)
                speed_percent = int((speed_factor - 1.0) * 100)
                if speed_percent > 0:
                    logger.info(f"Applied voice speed adjustment: +{speed_percent}% faster")
                else:
                    logger.info(f"Applied voice speed adjustment: {speed_percent}% slower")

            return audio_np.astype(np.float32)
        
        return None
    
    def _split_text_into_chunks(self, text: str, max_words: int = 250) -> List[str]:
        """Split long text into manageable chunks at sentence boundaries
        
        Args:
            text: The text to split
            max_words: Maximum words per chunk (default 250 for safety)
        
        Returns:
            List of text chunks
        """
        import re
        
        # Split into sentences (handling common abbreviations)
        # This regex tries to split on sentence endings while avoiding common abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # If regex split didn't work well, fall back to simple split
        if len(sentences) == 1 and len(text.split()) > max_words:
            # Fall back to splitting on any period followed by space
            sentences = text.replace('. ', '.|').split('|')
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If single sentence is too long, split it further
            if sentence_word_count > max_words:
                # Split long sentence at commas or semicolons
                sub_parts = re.split(r'[,;]', sentence)
                for part in sub_parts:
                    part = part.strip()
                    if not part:
                        continue
                    part_words = part.split()
                    part_word_count = len(part_words)
                    
                    if current_word_count + part_word_count > max_words and current_chunk:
                        # Save current chunk
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_word_count = part_word_count
                    else:
                        current_chunk.append(part)
                        current_word_count += part_word_count
            else:
                # Check if adding this sentence would exceed the limit
                if current_word_count + sentence_word_count > max_words and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If no chunks were created, return the original text
        if not chunks:
            chunks = [text]
        
        logger.info(f"Split text into {len(chunks)} chunks (max {max_words} words each)")
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            logger.debug(f"Chunk {i+1}: {word_count} words")
        
        return chunks
    
    def _parse_pause_keywords(self, text: str) -> List[Tuple[str, Any]]:
        """Parse [pause] and [pause:ms] keywords from text
        
        Args:
            text: Text potentially containing pause keywords
            
        Returns:
            List of tuples: ('text', str) or ('pause', duration_ms)
        """
        segments = []
        # Pattern matches [pause] or [pause:1500] where 1500 is milliseconds
        pattern = r'\[pause(?::(\d+))?\]'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            # Add text segment before pause (if any)
            if match.start() > last_end:
                text_segment = text[last_end:match.start()].strip()
                if text_segment:  # Only add non-empty text segments
                    segments.append(('text', text_segment))
            
            # Add pause segment with duration (default 1000ms = 1 second)
            duration_ms = int(match.group(1)) if match.group(1) else 1000
            segments.append(('pause', duration_ms))
            last_end = match.end()
        
        # Add remaining text after last pause (if any)
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                segments.append(('text', remaining_text))
        
        # If no pauses found, return original text as single segment
        if not segments:
            segments.append(('text', text))
        
        logger.debug(f"Parsed text into {len(segments)} segments (including pauses)")
        return segments
    
    def _generate_silence(self, duration_ms: int, sample_rate: int = 24000) -> dict:
        """Generate silence audio tensor for specified duration
        
        Args:
            duration_ms: Duration of silence in milliseconds
            sample_rate: Sample rate (default 24000 Hz for VibeVoice)
            
        Returns:
            Audio dict with silence waveform
        """
        # Calculate number of samples for the duration
        num_samples = int(sample_rate * duration_ms / 1000.0)
        
        # Create silence tensor with shape (1, 1, num_samples) to match audio format
        silence_waveform = torch.zeros(1, 1, num_samples, dtype=torch.float32)
        
        logger.info(f"Generated {duration_ms}ms silence ({num_samples} samples)")
        
        return {
            "waveform": silence_waveform,
            "sample_rate": sample_rate
        }
    
    def _format_text_for_vibevoice(self, text: str, speakers: list) -> str:
        """Format text with speaker information for VibeVoice using correct format"""
        # Remove any newlines from the text to prevent parsing issues
        # The processor splits by newline and expects each line to have "Speaker N:" format
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # VibeVoice expects format: "Speaker 1: text" not "Name: text"
        if len(speakers) == 1:
            return f"Speaker 1: {text}"
        else:
            # Check if text already has proper Speaker N: format
            if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
                return text
            # If text has name format, convert to Speaker N format
            elif any(f"{speaker}:" in text for speaker in speakers):
                formatted_text = text
                for i, speaker in enumerate(speakers):
                    formatted_text = formatted_text.replace(f"{speaker}:", f"Speaker {i+1}:")
                return formatted_text
            else:
                # Plain text, assign to first speaker
                return f"Speaker 1: {text}"
    
    def _generate_with_vibevoice(self, formatted_text: str, voice_samples: List[np.ndarray],
                                cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                                temperature: float = 0.95, top_p: float = 0.95, llm_lora_strength: float = 1.0) -> dict:
        """Generate audio using VibeVoice model"""
        try:
            # Ensure model and processor are loaded
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # Also set numpy seed for any numpy operations
            np.random.seed(seed)
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            logger.info(f"Starting audio generation with {diffusion_steps} diffusion steps...")
            
            # Check for interruption before starting generation
            if INTERRUPTION_SUPPORT:
                try:
                    import comfy.model_management as mm
                    
                    # Check if we're being interrupted right now
                    # The interrupt flag is reset by ComfyUI before each node execution
                    # So we only check model_management's throw_exception_if_processing_interrupted
                    # which is the proper way to check for interruption
                    mm.throw_exception_if_processing_interrupted()
                    
                except ImportError:
                    # If comfy.model_management is not available, skip this check
                    pass
            
            # Prepare inputs using processor
            inputs = self.processor(
                [formatted_text],  # Wrap text in list
                voice_samples=[voice_samples], # Provide voice samples for reference
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Estimate tokens for user information (not used as limit)
            text_length = len(formatted_text.split())
            estimated_tokens = int(text_length * 2.5)  # More accurate estimate for display
            
            # Log generation start with explanation
            logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
            logger.info(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            logger.info("The generation will stop automatically when audio is complete")
            
            # Create stop check function for interruption support
            stop_check_fn = None
            if INTERRUPTION_SUPPORT:
                def check_comfyui_interrupt():
                    """Check if ComfyUI has requested interruption"""
                    try:
                        if hasattr(execution, 'PromptExecutor') and hasattr(execution.PromptExecutor, 'interrupted'):
                            interrupted = execution.PromptExecutor.interrupted
                            if interrupted:
                                logger.info("Generation interrupted by user via stop_check_fn")
                            return interrupted
                    except:
                        pass
                    return False
                
                stop_check_fn = check_comfyui_interrupt
            
            # Generate with official parameters
            with torch.no_grad():
                if use_sampling:
                    # Use sampling mode (less stable but more varied)
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        stop_check_fn=stop_check_fn,
                    )
                else:
                    # Use deterministic mode like official examples
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=False,  # More deterministic generation
                        stop_check_fn=stop_check_fn,
                    )
                
                # Check if we got actual audio output
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    speech_tensors = output.speech_outputs

                    if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                        audio_tensor = torch.cat(speech_tensors, dim=-1)
                    else:
                        audio_tensor = speech_tensors
                    
                    # Ensure proper format (1, 1, samples)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    # Convert to float32 for compatibility with downstream nodes (Save Audio, etc.)
                    # Many audio processing nodes don't support BFloat16
                    return {
                        "waveform": audio_tensor.cpu().float(),
                        "sample_rate": 24000
                    }
                    
                elif hasattr(output, 'sequences'):
                    logger.error("VibeVoice returned only text tokens, no audio generated")
                    raise Exception("VibeVoice failed to generate audio - only text tokens returned")
                    
                else:
                    logger.error(f"Unexpected output format from VibeVoice: {type(output)}")
                    raise Exception(f"VibeVoice returned unexpected output format: {type(output)}")
                
        except Exception as e:
            # Re-raise interruption exceptions without wrapping
            import comfy.model_management as mm
            if isinstance(e, mm.InterruptProcessingException):
                raise  # Let the interruption propagate
            
            # For real errors, log and re-raise with context
            logger.error(f"VibeVoice generation failed: {e}")
            raise Exception(f"VibeVoice generation failed: {str(e)}")