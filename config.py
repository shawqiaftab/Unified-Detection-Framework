import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # disable TF GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # silence INFO logs

import random
import numpy as np
import tensorflow as tf
import torch
from pathlib import Path
import gc


GLOBAL_CONFIG = {
    # PATH
    'base_dir': '/kaggle/working/web_attack_detection',
    'data_dir': '/kaggle/working/cse-487-porjeckt', 
    'models_dir': '/kaggle/working/web_attack_detection/models',
    'results_dir': '/kaggle/working/web_attack_detection/results',
    'viz_dir': '/kaggle/working/web_attack_detection/visualizations',
    'logs_dir': '/kaggle/working/web_attack_detection/logs',

    # DATA SpLiT
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'stratify': True,

    # memory use
    'use_sample_data': False,
    'sample_fraction': 0.3,
    'max_samples': 50000,
    'clear_memory_after_model': True,
    'use_chunked_processing': True,
    'chunk_size': 5000,

    # =Training paraM
    'batch_size': 32,
    'epochs': 20,
    'early_stopping_patience': 5,
    'early_stopping_metric': 'val_loss',
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'binary_crossentropy',

    #  REGULARIZATION 
    'dropout_rate': 0.3,
    'l2_regularization': 0.01,

    #  CLASSICAL ML
    'cv_folds': 3,
    'max_iter': 500,
    'n_jobs': 2,

    # TRANSFORMER MODELS
    'max_seq_length': 64,
    'transformer_epochs': 1,
    'transformer_lr': 3e-5,
    'transformer_batch_size': 1,
    'gradient_accumulation_steps': 32,
    'warmup_ratio': 0.1,
    'fp16': True,
    'gradient_checkpointing': True,
    'max_grad_norm': 1.0,

    # FEATURE EXTRACTION 
    'word2vec_dim': 50,
    'fasttext_dim': 50,
    'use_dim': 512,
    'skip_use': False,
    'skip_bert': True,
    'skip_hybrid': False,
    'tfidf_max_features': 1000,
    'tfidf_ngram_range': (1, 2),

    # GNN 
    'node_feature_dim': 128,  # : 64
    'gnn_hidden_dims': [128, 64, 32], 
    'gnn_layers': 3,
    'gnn_dropout': 0.3,
    'gat_heads': 2,  # 8
    'gnn_epochs': 10,  # 20
    'gat_epochs': 5,   # 20
    'gnn_graphs_input_dir': '/kaggle/input/cse-487-porjeckt',
    'gnn_lr': 0.001,
    'gnn_batch_size': 32,  # 60

    # GNN specific
    'train_gnn': True,
    'gnn_sample_fraction': 0.05,  # 1.
    'gnn_max_train_samples': 6000,  # 
    'gnn_max_val_samples': 1200,  # 
    'gnn_max_test_samples': 1200,


    # HARDWARE
    'use_gpu': True,
    'mixed_precision': True,
    'num_workers': 2,
    'pin_memory': False,

    #  SAMPLE COLLECTION IF USED WITH XAI?
    'samples_per_category': 20,
    'save_lime_explanations': False,
    'lime_samples': 5,

    # 
    'train_classical_ml': True,
    'train_deep_learning': True,
    'train_transformers': True,
    'use_sampled_data_for_transformers': True,
    'transformer_sample_fraction': 0.1,
    'train_gnn': True,
    'train_hybrid': True,
}

# MODEL ARCHITECTURE SPECIFICATIONS

MODEL_ARCHITECTURES = {
    'MLP': {
        'layers': [256, 128, 64],
        'activations': ['relu', 'relu', 'relu'],
        'dropout': 0.3,
    },
    'CNN': {
        'conv_layers': [
            {'filters': 16, 'kernel_size': 3},
            {'filters': 32, 'kernel_size': 3},
        ],
        'pool_size': 2,
        'dense_units': 64,
        'dropout': 0.3,
    },
    'BiLSTM': {
        'lstm_units': [64, 32],
        'bidirectional': True,
        'return_sequences': [True, False],
        'dense_units': 64,
        'dropout': 0.3,
    },
    'CNN_LSTM': {
        'conv_layers': [
            {'filters': 32, 'kernel_size': 3},
            {'filters': 64, 'kernel_size': 3},
        ],
        'pool_size': 2,
        'lstm_units': 64,
        'dense_units': 32,
        'dropout': 0.3,
    },
}


PRIMARY_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
    'pr_auc', 'fpr', 'fnr', 'fp_count', 'fn_count', 'tp', 'tn', 'mcc'
]

TIMING_METRICS = [
    'training_time', 'inference_time_per_sample',
    'total_inference_time', 'throughput'
]

COMPUTATIONAL_METRICS = [
    'model_parameters', 'model_size_mb',
    'gpu_memory_mb', 'cpu_memory_mb'
]


ESSENTIAL_MODELS = [
    'Logistic_Regression', 'Random_Forest', 'XGBoost'
]

CLASSICAL_ML_MODELS = [
    'Logistic_Regression', 'SVM', 'Gaussian_Naive_Bayes',
    'Decision_Tree', 'KNN', 'Random_Forest', 'XGBoost',
    'Gradient_Boosting', 'Extra_Trees'
]

DEEP_LEARNING_MODELS = [
    'MLP', 'CNN', 'LSTM', 'BiLSTM', 'CNN_LSTM'
]

TRANSFORMER_MODELS = [
    'DistilBERT', 'BERT'
]

HYBRID_MODELS = [
    'Stacking_Ensemble', 'Soft_Voting', 'Hard_Voting'
]

GNN_MODELS = [
    'GCN', 'GAT'
]

ALL_MODELS = (CLASSICAL_ML_MODELS + DEEP_LEARNING_MODELS +
              TRANSFORMER_MODELS + HYBRID_MODELS + GNN_MODELS)


def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage():
    """Get current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    return mem_gb

def print_memory_status():
    """Print current memory status"""
    try:
        mem_gb = get_memory_usage()
        print(f"Current memory usage: {mem_gb:.2f} GB")
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            gpu_max = torch.cuda.max_memory_allocated() / 1024**3
            print(f"GPU memory: {gpu_mem:.2f} GB (max: {gpu_max:.2f} GB)")
    except:
        pass


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directory_structure():
    directories = [
        GLOBAL_CONFIG['base_dir'],
        GLOBAL_CONFIG['models_dir'],
        GLOBAL_CONFIG['results_dir'],
        GLOBAL_CONFIG['viz_dir'],
        GLOBAL_CONFIG['logs_dir'],
        f"{GLOBAL_CONFIG['models_dir']}/classical_ml",
        f"{GLOBAL_CONFIG['models_dir']}/deep_learning",
        f"{GLOBAL_CONFIG['models_dir']}/transformers",
        f"{GLOBAL_CONFIG['models_dir']}/hybrid",
        f"{GLOBAL_CONFIG['models_dir']}/gnn",
        f"{GLOBAL_CONFIG['results_dir']}/metrics",
        f"{GLOBAL_CONFIG['results_dir']}/predictions",
        f"{GLOBAL_CONFIG['results_dir']}/sample_outputs",
        f"{GLOBAL_CONFIG['results_dir']}/error_analysis",
        f"{GLOBAL_CONFIG['viz_dir']}/individual_models",
        f"{GLOBAL_CONFIG['viz_dir']}/comparative",
        f"{GLOBAL_CONFIG['logs_dir']}/training_logs",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Directory structure created successfully!")
    return directories

def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*70)
    print("GPU AVAILABILITY CHECK")
    print("="*70)

    print("\n TensorFlow:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"  Found GPU: {gpu}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("   Memory growth enabled")
        except RuntimeError as e:
            print(f"   Warning: {e}")
    else:
        print("   No GPU found, using CPU")

    print("\n PyTorch:")
    if torch.cuda.is_available():
        print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("   CUDA not available, using CPU")

    print("="*70 + "\n")
    print_memory_status()

def optimize_for_kaggle():
    """Apply Kaggle-specific optimizations"""
    print("\nApplying Kaggle optimizations...")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    print("Kaggle optimizations applied!")


if __name__ == "__main__":
    print("="*70)
    print("INITIALIZING PROJECT CONFIGURATION (KAGGLE OPTIMIZED)")
    print("="*70)

    set_seed(GLOBAL_CONFIG['random_seed'])
    create_directory_structure()
    optimize_for_kaggle()
    check_gpu()

    print("\nConfiguration initialized successfully!")
    print(f"Base directory: {GLOBAL_CONFIG['base_dir']}")
    print(f"Random seed: {GLOBAL_CONFIG['random_seed']}")
    print(f"\nMemory settings:")
    print(f" - Sample data: {GLOBAL_CONFIG['use_sample_data']}")
    print(f" - Batch size: {GLOBAL_CONFIG['batch_size']}")
    print(f" - TF-IDF features: {GLOBAL_CONFIG['tfidf_max_features']}")
    print(f" - Epochs: {GLOBAL_CONFIG['epochs']}")
