

from pathlib import Path
from typing import Dict, Any, List, Optional
import tensorflow as tf
from config.config import Config
from training.losses import get_loss_function
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
   "

    def __init__(self, model: tf.keras.Model, config: Config):
        """Initializes Trainer with target model and app configuration.

        Args:
            model: Keras feature embedding model instance.
            config: Config instance.
        """
        self.model = model
        self.config = config
        self.weights_dir = config.paths.weights_dir
        self.logs_dir = config.paths.logs_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _build_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Constructs Keras callbacks list for training monitoring."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if self.config.model.num_classes > 1 else "loss",
                patience=self.config.training.patience_early_stopping,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.weights_dir / "best_model.h5"),
                monitor="val_loss" if self.config.model.num_classes > 1 else "loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if self.config.model.num_classes > 1 else "loss",
                factor=0.5,
                patience=self.config.training.patience_reduce_lr,
                min_lr=self.config.training.min_learning_rate,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.logs_dir / "tensorboard"),
                histogram_freq=1,
                write_graph=True
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr * 0.95 if epoch > 10 else lr
            )
        ]
        return callbacks

    def compile_and_train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None
    ) -> tf.keras.callbacks.History:
        """Compiles model and runs fitting process.

        Args:
            train_dataset: Training tf.data.Dataset pipeline.
            val_dataset: Optional validation dataset.

        Returns:
            tf.keras.callbacks.History: Training metrics history.
        """
        loss_fn = get_loss_function(
            loss_name=self.config.training.loss_function,
            margin=self.config.training.margin,
            num_classes=self.config.model.num_classes
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.training.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["mae"] if "loss" in self.config.training.loss_function else ["accuracy"]
        )

        logger.info(f"Starting model training for {self.config.training.epochs} epochs...")

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.training.epochs,
            callbacks=self._build_callbacks(),
            verbose=1
        )

        logger.info("Training execution completed successfully.")
        return history
