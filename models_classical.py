
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import time
import json
from config import GLOBAL_CONFIG, set_seed, CLASSICAL_ML_MODELS



class ClassicalMLModelFactory:
    """Factory for creating classical ML models with standardized parameters"""
    
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
        
    def create_model(self, model_name):
        """Create model by name"""
        
        models = {
            'Logistic_Regression': LogisticRegression(
                max_iter=self.config['max_iter'],
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs']
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability estimates
                random_state=self.config['random_seed'],
                max_iter=self.config['max_iter']
            ),
            
            'Gaussian_Naive_Bayes': GaussianNB(),
            
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=10,
                criterion='gini',
                splitter='best',
                random_state=self.config['random_seed']
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='euclidean',
                n_jobs=self.config['n_jobs']
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                criterion='gini',
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs']
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs'],
                eval_metric='logloss'
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=self.config['random_seed']
            ),
            
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                criterion='gini',
                random_state=self.config['random_seed'],
                n_jobs=self.config['n_jobs']
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name]


class ClassicalMLTrainer:
    """Train and evaluate classical ML models"""
    
    def __init__(self, config=GLOBAL_CONFIG):
        self.config = config
        self.factory = ClassicalMLModelFactory(config)
        
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None):
        """
        Train a classical ML model
        
        Returns:
            Trained model, training metrics
        """
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}\n")
        
        model = self.factory.create_model(model_name)
        
        print(f" Training {model_name}...")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f" Training complete in {training_time:.2f} seconds")
        
        model_info = {
            'model_name': model_name,
            'training_time': training_time,
            'training_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
        }
        
        if hasattr(model, 'n_features_in_'):
            model_info['n_features_in'] = model.n_features_in_
        
        if hasattr(model, 'feature_importances_'):
            model_info['has_feature_importance'] = True
        
        return model, model_info
    
    def save_model(self, model, model_name, model_info, save_dir=None):
        """Save trained model and metadata"""
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/classical_ml"
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = f"{save_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        
 
        metadata_path = f"{save_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f" Model saved: {model_path}")
        print(f" Metadata saved: {metadata_path}")
        
    def load_model(self, model_name, save_dir=None):
        """Load trained model"""
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/classical_ml"
        
        model_path = f"{save_dir}/{model_name}.pkl"
        model = joblib.load(model_path)
        
        metadata_path = f"{save_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def train_all_classical_models(self, X_train, y_train, X_val, y_val, 
                                   feature_type='uniembed'):
        """
        Train all classical ML models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            feature_type: Type of features being used
            
        Returns:
            Dictionary of trained models and their info
        """
        print(f"\n{'='*70}")
        print(f"TRAINING ALL CLASSICAL ML MODELS")
        print(f"Feature Type: {feature_type.upper()}")
        print(f"{'='*70}\n")
        
        trained_models = {}
        
        for model_name in CLASSICAL_ML_MODELS:
            try:
                model, model_info = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                model_info['feature_type'] = feature_type
                
                self.save_model(model, model_name, model_info)
                
                trained_models[model_name] = {
                    'model': model,
                    'info': model_info
                }
                
                print(f" {model_name} complete!\n")
                
            except Exception as e:
                print(f" Error training {model_name}: {str(e)}\n")
                continue
        
        print(f"\n{'='*70}")
        print(f" ALL CLASSICAL ML MODELS TRAINED: {len(trained_models)}/{len(CLASSICAL_ML_MODELS)}")
        print(f"{'='*70}\n")
        
        return trained_models


if __name__ == "__main__":
    
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    feature_dir = f"{GLOBAL_CONFIG['base_dir']}/features"
    
    #  UniEmbed 
    X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
    X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
    X_test = np.load(f"{feature_dir}/X_test_uniembed.npy")
    

    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")

    trainer = ClassicalMLTrainer(GLOBAL_CONFIG)
    trained_models = trainer.train_all_classical_models(
        X_train, y_train, X_val, y_val, feature_type='uniembed'
    )
    
    print(" Classical ML training complete!")

