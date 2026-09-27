

from typing import Tuple, Optional
import numpy as np
import cv2
import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
  

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
       
        self.model = model
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            self.layer_name = layer_name

        logger.info(f"Initialized GradCAM targeting convolutional layer: '{self.layer_name}'")

    def _find_last_conv_layer(self) -> str:
      
        for layer in reversed(self.model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return layer.name
            # Handle nested functional backbones like EfficientNet
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        return sub_layer.name
        raise ValueError("No Conv2D convolutional layer found in target model.")

    def compute_heatmap(
        self,
        image_tensor: tf.Tensor,
        pred_index: Optional[int] = None
    ) -> np.ndarray:
       
        grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)
            if pred_index is None:
                # Target maximum activation vector component
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Extract gradient of class score w.r.t convolutional feature map
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation and normalize to range [0, 1]
        heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        
        # Resize heatmap to original image resolution
        heatmap_resized = cv2.resize(heatmap, (original_image_rgb.shape[1], original_image_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply COLORMAP_JET
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        # Superimpose heatmap
        superimposed = color_heatmap * alpha + original_image_rgb * (1.0 - alpha)
        return np.clip(superimposed, 0, 255).astype(np.uint8)
