
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU training for DL models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TF warnings

import numpy as np
import tensorflow as tf
from config import GLOBAL_CONFIG, DEEP_LEARNING_MODELS, MODEL_ARCHITECTURES, clear_memory

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time
import json
from pathlib import Path
import gc


class DeepLearningModelFactory:
    """Factory for creating deep learning models"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        
    def create_mlp(self, input_dim):
        """Multi-Layer Perceptron"""
        arch = MODEL_ARCHITECTURES['MLP']
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(arch['layers'][0], activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization'])),
            layers.Dropout(arch['dropout']),
            layers.Dense(arch['layers'][1], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization'])),
            layers.Dropout(arch['dropout']),
            layers.Dense(arch['layers'][2], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization'])),
            layers.Dropout(arch['dropout']),
            layers.Dense(1, activation='sigmoid')
        ], name='MLP')
        
        return model
    
    def create_cnn(self, input_dim):
        """Convolutional Neural Network"""
        arch = MODEL_ARCHITECTURES['CNN']
        
        inputs = layers.Input(shape=(input_dim, 1))
        
        x = layers.Conv1D(arch['conv_layers'][0]['filters'], 
                         kernel_size=arch['conv_layers'][0]['kernel_size'], 
                         activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=arch['pool_size'])(x)
        
        x = layers.Conv1D(arch['conv_layers'][1]['filters'], 
                         kernel_size=arch['conv_layers'][1]['kernel_size'], 
                         activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=arch['pool_size'])(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(arch['dense_units'], activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.config['l2_regularization']))(x)
        x = layers.Dropout(arch['dropout'])(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN')
        return model
    
    def create_lstm(self, input_dim, timesteps=None):
        """LSTM Network"""
        if timesteps is None:
            timesteps = min(input_dim, 50)
        
        arch = MODEL_ARCHITECTURES['BiLSTM']
        
        inputs = layers.Input(shape=(timesteps, input_dim // timesteps))
        
        x = layers.LSTM(arch['lstm_units'][0], return_sequences=True)(inputs)
        x = layers.Dropout(arch['dropout'])(x)
        x = layers.LSTM(arch['lstm_units'][1])(x)
        x = layers.Dropout(arch['dropout'])(x)
        x = layers.Dense(arch['dense_units'], activation='relu')(x)
        x = layers.Dropout(arch['dropout'])(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
        return model
    
    def create_bilstm(self, input_dim, timesteps=None):
        """Bidirectional LSTM"""
        if timesteps is None:
            timesteps = min(input_dim, 50)
        
        arch = MODEL_ARCHITECTURES['BiLSTM']
        
        inputs = layers.Input(shape=(timesteps, input_dim // timesteps))
        
        x = layers.Bidirectional(layers.LSTM(arch['lstm_units'][0], return_sequences=True))(inputs)
        x = layers.Dropout(arch['dropout'])(x)
        x = layers.Bidirectional(layers.LSTM(arch['lstm_units'][1]))(x)
        x = layers.Dropout(arch['dropout'])(x)
        x = layers.Dense(arch['dense_units'], activation='relu')(x)
        x = layers.Dropout(arch['dropout'])(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='BiLSTM')
        return model
    
    def create_cnn_lstm(self, input_dim, timesteps=None):
        """CNN-LSTM Hybrid"""
        if timesteps is None:
            timesteps = min(input_dim, 50)
        
        arch = MODEL_ARCHITECTURES['CNN_LSTM']
        
        inputs = layers.Input(shape=(timesteps, input_dim // timesteps))
        
        x = layers.Conv1D(arch['conv_layers'][0]['filters'], 
                         kernel_size=arch['conv_layers'][0]['kernel_size'], 
                         activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=arch['pool_size'])(x)
        x = layers.Conv1D(arch['conv_layers'][1]['filters'], 
                         kernel_size=arch['conv_layers'][1]['kernel_size'], 
                         activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=arch['pool_size'])(x)
        
        x = layers.LSTM(arch['lstm_units'])(x)
        x = layers.Dropout(arch['dropout'])(x)
        
        x = layers.Dense(arch['dense_units'], activation='relu')(x)
        x = layers.Dropout(arch['dropout'])(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
        return model
    
    def create_model(self, model_name, input_dim):
        """Create model by name"""
        
        creators = {
            'MLP': self.create_mlp,
            'CNN': self.create_cnn,
            'LSTM': self.create_lstm,
            'BiLSTM': self.create_bilstm,
            'CNN_LSTM': self.create_cnn_lstm
        }
        
        if model_name not in creators:
            raise ValueError(f"Unknown model: {model_name}")
        
        return creators[model_name](input_dim)



class DeepLearningTrainer:
    """Train and evaluate deep learning models"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.factory = DeepLearningModelFactory(config)
        
    def prepare_data_for_model(self, X, model_name):
        """Reshape data based on model requirements"""
        
        if model_name == 'MLP':
            return X
        
        elif model_name == 'CNN':
            return X.reshape(X.shape[0], X.shape[1], 1)
        
        elif model_name in ['LSTM', 'BiLSTM', 'CNN_LSTM']:
            n_features = X.shape[1]
            timesteps = min(n_features, 50)
            features_per_timestep = n_features // timesteps
            
            truncated_features = timesteps * features_per_timestep
            X_truncated = X[:, :truncated_features]
            
            return X_truncated.reshape(X.shape[0], timesteps, features_per_timestep)
        
        else:
            return X
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train a deep learning model"""
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}\n")
        
        print("Preparing data...")
        X_train_prepared = self.prepare_data_for_model(X_train, model_name)
        X_val_prepared = self.prepare_data_for_model(X_val, model_name)
        
        print(f"   Train shape: {X_train_prepared.shape}")
        print(f"   Val shape: {X_val_prepared.shape}")
        
        input_dim = X_train.shape[1]
        model = self.factory.create_model(model_name, input_dim)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=self.config['loss_function'],
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"\nModel Architecture:")
        model.summary()
        
        callbacks = [
            EarlyStopping(
                monitor=self.config['early_stopping_metric'],
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        history = model.fit(
            X_train_prepared, y_train,
            validation_data=(X_val_prepared, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining complete in {training_time:.2f} seconds")
        
        model_info = {
            'model_name': model_name,
            'training_time': training_time,
            'total_parameters': model.count_params(),
            'input_shape': str(X_train_prepared.shape[1:]),
            'epochs_trained': len(history.history['loss']),
            'best_val_loss': float(np.min(history.history['val_loss'])),
            'best_val_accuracy': float(np.max(history.history['val_accuracy'])),
        }
        
        return model, history, model_info
    
    def save_model(self, model, model_name, model_info, history, save_dir=None):
        """Save trained model"""
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/deep_learning"
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = f"{save_dir}/{model_name}.h5"
        model.save(model_path)
        print(f"Model saved: {model_path}")

        # Keras >= 2.12 requires the weights file to end in .weights.h5
        weights_path = f"{save_dir}/{model_name}.weights.h5"
        model.save_weights(weights_path)
        print(f"Weights saved: {weights_path}")
        
        metadata_path = f"{save_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
        
        history_path = f"{save_dir}/{model_name}_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"History saved: {history_path}")
    
    def train_all_deep_learning_models(self, X_train, y_train, X_val, y_val, 
                                       feature_type='uniembed'):
        """Train all deep learning models"""
        
        print(f"\n{'='*70}")
        print(f"TRAINING ALL DEEP LEARNING MODELS")
        print(f"Feature Type: {feature_type.upper()}")
        print(f"{'='*70}\n")
        
        trained_models = {}
        
        for model_name in DEEP_LEARNING_MODELS:
            try:
                model, history, model_info = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                model_info['feature_type'] = feature_type
                
                self.save_model(model, model_name, model_info, history)
                
                trained_models[model_name] = {
                    'model': model,
                    'history': history,
                    'info': model_info
                }
                
                print(f"{model_name} complete!\n")
                
                # MEMORY OPTIMIZATION: Clear memory
                del model
                keras.backend.clear_session()
                gc.collect()
                
                if self.config.get('clear_memory_after_model', True):
                    clear_memory()
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}\n")
                keras.backend.clear_session()
                continue
        
        print(f"\n{'='*70}")
        print(f"ALL DEEP LEARNING MODELS TRAINED: {len(trained_models)}/{len(DEEP_LEARNING_MODELS)}")
        print(f"{'='*70}\n")
        
        return trained_models


if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    feature_dir = f"{GLOBAL_CONFIG['base_dir']}/features"
    X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
    X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
    X_test = np.load(f"{feature_dir}/X_test_uniembed.npy")
    
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    trainer = DeepLearningTrainer(GLOBAL_CONFIG)
    trained_models = trainer.train_all_deep_learning_models(
        X_train, y_train, X_val, y_val, feature_type='uniembed'
    )
    
    print("Deep learning training complete!")

