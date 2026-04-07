
import os
import time
import json
from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow import keras

from config import GLOBAL_CONFIG, set_seed, HYBRID_MODELS


class LightGBM_BiLSTM_Hybrid:
    """
    Two-tier hybrid model:
    - Tier 1: LightGBM handles high-confidence predictions
    - Tier 2: BiLSTM processes ambiguous cases
    """
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
        self.confidence_threshold = 0.9
        self.lightgbm_model = None
        self.bilstm_model = None

    def create_lightgbm(self):
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=self.config['random_seed'],
            n_jobs=self.config['n_jobs']
        )

    def create_bilstm(self, input_dim):
        timesteps = min(input_dim, 50)
        features_per_timestep = input_dim // timesteps

        model = keras.Sequential([
            keras.layers.Input(shape=(timesteps, features_per_timestep)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dropout(0.3),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ], name='BiLSTM_Tier2')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_bilstm_data(self, X):
        n_features = X.shape[1]
        timesteps = min(n_features, 50)
        features_per_timestep = n_features // timesteps
        truncated_features = timesteps * features_per_timestep
        X_truncated = X[:, :truncated_features]
        return X_truncated.reshape(X.shape[0], timesteps, features_per_timestep)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("\nTraining Tier 1: LightGBM...")
        self.lightgbm_model = self.create_lightgbm()
        self.lightgbm_model.fit(X_train, y_train)

        train_proba = self.lightgbm_model.predict_proba(X_train)
        train_confidence = np.max(train_proba, axis=1)

        ambiguous_mask = train_confidence < self.confidence_threshold
        X_train_ambiguous = X_train[ambiguous_mask]
        y_train_ambiguous = y_train[ambiguous_mask]

        print(f"   High-confidence samples: {np.sum(~ambiguous_mask)} ({np.sum(~ambiguous_mask)/len(y_train)*100:.1f}%)")
        print(f"   Ambiguous samples: {np.sum(ambiguous_mask)} ({np.sum(ambiguous_mask)/len(y_train)*100:.1f}%)")

        if np.sum(ambiguous_mask) > 0:
            print("\nTraining Tier 2: BiLSTM on ambiguous cases...")
            self.bilstm_model = self.create_bilstm(X_train.shape[1])
            X_train_bilstm = self.prepare_bilstm_data(X_train_ambiguous)

            if X_val is not None and y_val is not None:
                val_proba = self.lightgbm_model.predict_proba(X_val)
                val_confidence = np.max(val_proba, axis=1)
                val_ambiguous_mask = val_confidence < self.confidence_threshold

                if np.sum(val_ambiguous_mask) > 0:
                    X_val_ambiguous = X_val[val_ambiguous_mask]
                    y_val_ambiguous = y_val[val_ambiguous_mask]
                    X_val_bilstm = self.prepare_bilstm_data(X_val_ambiguous)
                    self.bilstm_model.fit(
                        X_train_bilstm, y_train_ambiguous,
                        validation_data=(X_val_bilstm, y_val_ambiguous),
                        epochs=self.config['epochs'],
                        batch_size=self.config['batch_size'],
                        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                        verbose=1
                    )
                else:
                    self.bilstm_model.fit(
                        X_train_bilstm, y_train_ambiguous,
                        epochs=self.config['epochs'],
                        batch_size=self.config['batch_size'],
                        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                        verbose=1
                    )
            else:
                self.bilstm_model.fit(
                    X_train_bilstm, y_train_ambiguous,
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    verbose=1
                )

        print("Two-tier hybrid model trained!")
        return self

    def predict(self, X):
        predictions = np.zeros(len(X))
        lgbm_proba = self.lightgbm_model.predict_proba(X)
        lgbm_confidence = np.max(lgbm_proba, axis=1)
        lgbm_predictions = self.lightgbm_model.predict(X)

        high_conf_mask = lgbm_confidence >= self.confidence_threshold
        predictions[high_conf_mask] = lgbm_predictions[high_conf_mask]

        if self.bilstm_model is not None and np.sum(~high_conf_mask) > 0:
            X_ambiguous = X[~high_conf_mask]
            X_ambiguous_bilstm = self.prepare_bilstm_data(X_ambiguous)
            bilstm_proba = self.bilstm_model.predict(X_ambiguous_bilstm, verbose=0)
            predictions[~high_conf_mask] = (bilstm_proba.flatten() > 0.5).astype(int)
        else:
            predictions[~high_conf_mask] = lgbm_predictions[~high_conf_mask]
        return predictions.astype(int)

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        lgbm_proba = self.lightgbm_model.predict_proba(X)
        lgbm_confidence = np.max(lgbm_proba, axis=1)
        high_conf_mask = lgbm_confidence >= self.confidence_threshold
        proba[high_conf_mask] = lgbm_proba[high_conf_mask]

        if self.bilstm_model is not None and np.sum(~high_conf_mask) > 0:
            X_ambiguous = X[~high_conf_mask]
            X_ambiguous_bilstm = self.prepare_bilstm_data(X_ambiguous)
            bilstm_proba = self.bilstm_model.predict(X_ambiguous_bilstm, verbose=0).flatten()
            proba[~high_conf_mask, 1] = bilstm_proba
            proba[~high_conf_mask, 0] = 1 - bilstm_proba
        else:
            proba[~high_conf_mask] = lgbm_proba[~high_conf_mask]
        return proba

class StackingEnsembleFactory:
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with sklearn compatibility"""
        
        try:
            base_learners = [
                ('xgboost', XGBClassifier(
                    n_estimators=300,  
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.config['random_seed'],
                    n_jobs=self.config['n_jobs'],
                    eval_metric='logloss'
                )),
                ('random_forest', RandomForestClassifier(
                    n_estimators=100,  
                    max_depth=10,
                    random_state=self.config['random_seed'],
                    n_jobs=self.config['n_jobs']
                )),
                ('extra_trees', ExtraTreesClassifier(
                    n_estimators=100,  
                    max_depth=10,
                    random_state=self.config['random_seed'],
                    n_jobs=self.config['n_jobs']
                ))
            ]
            
            meta_learner = LogisticRegression(
                max_iter=500,
                random_state=self.config['random_seed'],
                n_jobs=2  
            )
            
            stacking_model = StackingClassifier(
                estimators=base_learners,
                final_estimator=meta_learner,
                cv=5,  
                n_jobs=1,  
                passthrough=False
            )
            
            return stacking_model
            
        except Exception as e:
            print(f"Warning: Could not create stacking ensemble: {e}")
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs']
            )


class VotingEnsembleFactory:
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
    
    def create_voting_ensemble(self, voting='soft'):
        """Create voting ensemble with sklearn compatibility"""
        
        try:
            estimators = [
                ('logistic', LogisticRegression(
                    max_iter=500,
                    random_state=self.config['random_seed'],
                    n_jobs=1
                )),
                ('random_forest', RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=self.config['random_seed'],
                    n_jobs=self.config['n_jobs']
                )),
                ('xgboost', XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=self.config['random_seed'],
                    n_jobs=self.config['n_jobs'],
                    eval_metric='logloss'
                ))
            ]
            
            voting_model = VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=1  
            )
            
            return voting_model
            
        except Exception as e:
            print(f"Warning: Could not create voting ensemble: {e}")
            # Fallback: return just Random Forest
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs']
            )


class BERT_XGBoost_Hybrid:
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
        self.xgboost_model = None
    
    def fit(self, X_bert_embeddings, y_train):
        print("\nTraining BERT-XGBoost Hybrid...")
        print("   Using pre-computed BERT embeddings as features")
        self.xgboost_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config['random_seed'],
            n_jobs=self.config['n_jobs'],
            eval_metric='logloss'
        )
        self.xgboost_model.fit(X_bert_embeddings, y_train)
        print("BERT-XGBoost Hybrid trained!")
        return self
    
    def predict(self, X_bert_embeddings):
        return self.xgboost_model.predict(X_bert_embeddings)
    
    def predict_proba(self, X_bert_embeddings):
        return self.xgboost_model.predict_proba(X_bert_embeddings)


class HybridModelTrainer:
    """Train and evaluate hybrid ensemble models"""
    
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
    
    def train_hybrid_model(self, model_name, X_train, y_train, X_val=None, y_val=None, 
                           X_bert_train=None, X_bert_val=None):
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if model_name == 'LightGBM_BiLSTM':
            model = LightGBM_BiLSTM_Hybrid(self.config)
            model.fit(X_train, y_train, X_val, y_val)
            
        elif model_name == 'Stacking_Ensemble':
            factory = StackingEnsembleFactory(self.config)
            model = factory.create_stacking_ensemble()
            print("Training stacking ensemble (this may take a while)...")
            model.fit(X_train, y_train)
            
        elif model_name == 'Soft_Voting':
            factory = VotingEnsembleFactory(self.config)
            model = factory.create_voting_ensemble(voting='soft')
            print("Training soft voting ensemble...")
            model.fit(X_train, y_train)
            
        elif model_name == 'Hard_Voting':
            factory = VotingEnsembleFactory(self.config)
            model = factory.create_voting_ensemble(voting='hard')
            print("Training hard voting ensemble...")
            model.fit(X_train, y_train)
            
        elif model_name == 'BERT_XGBoost':
            if X_bert_train is None:
                raise ValueError("BERT embeddings required for BERT_XGBoost model")
            model = BERT_XGBoost_Hybrid(self.config)
            model.fit(X_bert_train, y_train)
            
        else:
            raise ValueError(f"Unknown hybrid model: {model_name}")
        
        training_time = time.time() - start_time
        print(f"Training complete in {training_time:.2f} seconds")
        
        model_info = {
            'model_name': model_name,
            'training_time': training_time,
            'training_samples': X_train.shape[0],
        }
        return model, model_info
    
    def save_model(self, model, model_name, model_info, save_dir=None):
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/hybrid"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = f"{save_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")
        
        metadata_path = f"{save_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
    
    def train_all_hybrid_models(self, X_train, y_train, X_val, y_val, 
                                X_bert_train=None, X_bert_val=None, models_override=None):
        print(f"\n{'='*70}")
        print(f"TRAINING ALL HYBRID MODELS")
        print(f"{'='*70}\n")
        
        trained_models = {}
        model_list = models_override if models_override is not None else HYBRID_MODELS
        
        for model_name in model_list:
            try:
                if model_name == 'BERT_XGBoost' and X_bert_train is None:
                    print("Skipping BERT_XGBoost: BERT embeddings not provided.")
                    continue
                
                model, model_info = self.train_hybrid_model(
                    model_name, X_train, y_train, X_val, y_val,
                    X_bert_train=X_bert_train, X_bert_val=X_bert_val
                )
                self.save_model(model, model_name, model_info)
                trained_models[model_name] = {'model': model, 'info': model_info}
                print(f"{model_name} complete!\n")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}\n")
                continue
        
        print(f"\n{'='*70}")
        print(f"ALL HYBRID MODELS TRAINED: {len(trained_models)}/{len(model_list)}")
        print(f"{'='*70}\n")
        return trained_models


if __name__ == "__main__":
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    feature_dir = f"{GLOBAL_CONFIG['base_dir']}/features"
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    
    X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
    X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    
    # Optional BERT embeddings
    X_bert_train = X_bert_val = None
    bert_train_path = f"{feature_dir}/X_train_bert.npy"
    bert_val_path = f"{feature_dir}/X_val_bert.npy"
    if os.path.exists(bert_train_path) and os.path.exists(bert_val_path):
        X_bert_train = np.load(bert_train_path)
        X_bert_val = np.load(bert_val_path)
        print("Loaded BERT embeddings for hybrids.")
    else:
        print("BERT embeddings not found. BERT_XGBoost will be skipped.")
    
    trainer = HybridModelTrainer(GLOBAL_CONFIG)
    trained_models = trainer.train_all_hybrid_models(
        X_train, y_train, X_val, y_val,
        X_bert_train=X_bert_train, X_bert_val=X_bert_val
    )
    
    print("Hybrid model training complete!")

