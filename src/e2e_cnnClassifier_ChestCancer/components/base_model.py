import os
import tensorflow as tf
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from e2e_cnnClassifier_ChestCancer.entity.config_entity import BaseModelConfigAndParams

class BaseModel:
    def __init__(self, config: BaseModelConfigAndParams):
        self.config = config
        self.model = None  # Placeholder for the base model

    def download_and_save_base_model(self):
        """Downloads the base model and saves it to the specified path."""
        self.model = tf.keras.applications.EfficientNetB0(
            weights=self.config.training_weights,
            include_top=False,
            input_shape=self.config.model_input_shape
        )
        
        self.save_model(path=Path(self.config.base_model_path), model=self.model)

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model, classes: int, freeze_all: bool, freeze_till: int, learning_rate: float) -> tf.keras.Model:
        """Prepares the full model by adding a custom classifier on top of the base model."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),  
            metrics=["accuracy"] 
        )

        return full_model

    def update_base_model(self):
        """Updates the base model by preparing the full model and saving it."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.model_num_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.training_learning_rate
        )

        self.save_model(path=Path(self.config.updated_model_path), model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the specified path."""
        model.save(path)
