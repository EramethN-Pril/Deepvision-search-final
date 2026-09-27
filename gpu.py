

from typing import Dict, Any, List
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_gpu_environment(enable_mixed_precision: bool = True) -> Dict[str, Any]:
   
    gpus: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")
    gpu_status = {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_devices": [],
        "mixed_precision_enabled": False
    }

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                gpu_status["gpu_devices"].append(gpu.name)
            logger.info(f"Successfully configured memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            logger.warning(f"Error configuring GPU memory growth: {e}")

        if enable_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                gpu_status["mixed_precision_enabled"] = True
                logger.info("Mixed precision policy 'mixed_float16' enabled.")
            except Exception as e:
                logger.warning(f"Failed to set mixed precision policy: {e}")
    else:
        logger.info("No physical GPU detected. Running on CPU.")

    return gpu_status


def get_device_info() -> Dict[str, Any]:
   
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    return {
        "num_cpus": len(cpus),
        "num_gpus": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus],
        "tf_version": tf.__version__
    }
