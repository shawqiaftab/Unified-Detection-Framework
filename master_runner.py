
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import os
import pickle

from config import (
    GLOBAL_CONFIG, 
    set_seed, 
    create_directory_structure, 
    check_gpu,
    CLASSICAL_ML_MODELS,
    DEEP_LEARNING_MODELS,
    TRANSFORMER_MODELS,
    HYBRID_MODELS,
    GNN_MODELS
)

class MasterPipeline:
    """Master pipeline for training all models"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        
    def print_step(self, step, total, description):
        """Print formatted step header"""
        print(f"\n[STEP {step}/{total}] {description}")
        print("-"*70)
    
    def run_setup(self):
        """Step 1: Setup and configuration"""
        self.print_step(1, 10, "SETUP AND CONFIGURATION")
        set_seed(self.config['random_seed'])
        create_directory_structure()
        check_gpu()
        print("Setup complete!")
    
    def run_data_preprocessing(self):
        """Step 2: Data preprocessing — source-aware splits (no leakage)"""
        self.print_step(2, 10, "DATA PREPROCESSING")

        from data_preprocessing import DataLoader
        import json

        loader = DataLoader(data_dir=self.config['data_dir'], config=self.config)

        splits, source_meta = loader.load_and_split()

        data_dir = f"{self.config['base_dir']}/data"
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        for key, val in splits.items():
            np.save(f"{data_dir}/{key}.npy", val)

        # Save source metadata so evaluation can report per-source breakdown
        with open(f"{data_dir}/source_meta.json", 'w') as f:
            json.dump(source_meta, f, indent=2)

        print("\nSplit sizes:")
        for k, v in splits.items():
            print(f"  {k}: {v.shape}")
        print("\nSource breakdown saved to source_meta.json")
        print("Data preprocessing complete!")
        return splits
    
    def run_feature_extraction(self, splits=None):
        """Step 3: Feature extraction — train/val/test + cross-dataset split"""
        self.print_step(3, 10, "FEATURE EXTRACTION")

        from feature_engineering import extract_all_features, extract_cross_features

        if splits is None:
            data_dir = f"{self.config['base_dir']}/data"
            splits = {
                'X_train': np.load(f"{data_dir}/X_train.npy", allow_pickle=True),
                'y_train': np.load(f"{data_dir}/y_train.npy"),
                'X_val':   np.load(f"{data_dir}/X_val.npy",   allow_pickle=True),
                'y_val':   np.load(f"{data_dir}/y_val.npy"),
                'X_test':  np.load(f"{data_dir}/X_test.npy",  allow_pickle=True),
                'y_test':  np.load(f"{data_dir}/y_test.npy"),
            }

        features = extract_all_features(splits, self.config)

        extract_cross_features(self.config)

        print("Feature extraction complete!")
        return features
    
    def run_classical_ml_training(self):
        """Step 4: Train classical ML models"""
        self.print_step(4, 10, "TRAINING CLASSICAL ML MODELS")
        
        from models_classical import ClassicalMLTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        X_train = np.load(f"{feature_dir}/X_train_tfidf.npy")
        X_val = np.load(f"{feature_dir}/X_val_tfidf.npy")
        
        data_dir = f"{self.config['base_dir']}/data"
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = ClassicalMLTrainer(self.config)
        trained_models = trainer.train_all_classical_models(
            X_train, y_train, X_val, y_val, feature_type='tfidf'
        )
        
        print(f"Classical ML training complete! Trained {len(trained_models)} models.")
    
    def run_deep_learning_training(self):
        """Step 5: Train deep learning models"""
        self.print_step(5, 10, "TRAINING DEEP LEARNING MODELS")
        
        from models_deep_learning import DeepLearningTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
        X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
        
        data_dir = f"{self.config['base_dir']}/data"
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = DeepLearningTrainer(self.config)
        trained_models = trainer.train_all_deep_learning_models(
            X_train, y_train, X_val, y_val, feature_type='uniembed'
        )
        
        print(f"Deep learning training complete! Trained {len(trained_models)} models.")
    
    def run_transformer_training(self):
        """Step 6: Train transformer models"""
        self.print_step(6, 10, "FINE-TUNING TRANSFORMER MODELS")
        
        from models_transformers import TransformerTrainer
        
        data_dir = f"{self.config['base_dir']}/data"
        X_train = np.load(f"{data_dir}/X_train.npy", allow_pickle=True)
        X_val = np.load(f"{data_dir}/X_val.npy", allow_pickle=True)
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        trainer = TransformerTrainer(self.config)
        trained_models = trainer.train_all_transformer_models(
            X_train, y_train, X_val, y_val
        )
        
        print(f"Transformer training complete! Trained {len(trained_models)} models.")
    
    def run_gnn_training(self):
        """Step 7: Train GNN models"""
        self.print_step(7, 10, "TRAINING GRAPH NEURAL NETWORK MODELS")
        
        from models_gnn import GraphConstructor, GNNTrainer
        from gensim.models import Word2Vec
        import pickle
        
        feature_dir = f"{self.config['base_dir']}/features"
        word2vec_model = Word2Vec.load(f"{feature_dir}/word2vec.model")
        
        data_dir = f"{self.config['base_dir']}/data"
        X_train = np.load(f"{data_dir}/X_train.npy", allow_pickle=True)
        X_val = np.load(f"{data_dir}/X_val.npy", allow_pickle=True)
        X_test = np.load(f"{data_dir}/X_test.npy", allow_pickle=True)
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        y_test = np.load(f"{data_dir}/y_test.npy")
        

        if self.config.get('gnn_use_sampling', False):
            print("\nGNN sampling enabled (gnn_use_sampling=True in config).")
            print("WARNING: Using fewer samples than other models weakens comparability.")

            rng = np.random.default_rng(self.config['random_seed'])

            def stratified_cap(X, y, max_n):
                """Stratified cap so class balance is preserved."""
                classes = np.unique(y)
                per_class = max_n // len(classes)
                indices = []
                for cls in classes:
                    cls_idx = np.where(y == cls)[0]
                    n = min(per_class, len(cls_idx))
                    indices.append(rng.choice(cls_idx, n, replace=False))
                return np.concatenate(indices)

            tr_idx = stratified_cap(X_train, y_train, self.config['gnn_max_train_samples'])
            X_train, y_train = X_train[tr_idx], y_train[tr_idx]

            va_idx = stratified_cap(X_val, y_val, self.config['gnn_max_val_samples'])
            X_val, y_val = X_val[va_idx], y_val[va_idx]

            te_idx = stratified_cap(X_test, y_test, self.config['gnn_max_test_samples'])
            X_test, y_test = X_test[te_idx], y_test[te_idx]
        else:
            print("\nGNN using full dataset splits (same as all other models).")

        print(f"\nGNN Dataset sizes:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val:   {len(X_val)}")
        print(f"  Test:  {len(X_test)}\n")
        
        import gc
        import torch

        files_to_purge = [
            (feature_dir, 'X_train_tfidf.npy'),    (feature_dir, 'X_val_tfidf.npy'),
        ]
        print("\nFreeing files from disk to make room for graph files...")
        for d, fname in files_to_purge:
            fpath = f"{d}/{fname}"
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"  Deleted {fname}")
        gc.collect()

        constructor = GraphConstructor(word2vec_model)


        kaggle_input_graphs = self.config.get('gnn_graphs_input_dir', None)

        train_pt_input = f"{kaggle_input_graphs}/train_graphs.pt" if kaggle_input_graphs else None
        val_pt_input   = f"{kaggle_input_graphs}/val_graphs.pt"   if kaggle_input_graphs else None

        if train_pt_input and os.path.exists(train_pt_input) and os.path.exists(val_pt_input):
            print(f"Loading pre-built graphs from {kaggle_input_graphs} ...")
            train_graphs = torch.load(train_pt_input, weights_only=False)
            val_graphs   = torch.load(val_pt_input,   weights_only=False)
            print(f"  Loaded {len(train_graphs)} train + {len(val_graphs)} val graphs.")
            del X_train, X_val
            gc.collect()

            print("Constructing test graphs...")
            test_graphs = constructor.texts_to_graphs(X_test, y_test)
            del X_test
            gc.collect()
        else:
            print("Constructing training graphs...")
            train_graphs = constructor.texts_to_graphs(X_train, y_train)
            del X_train
            gc.collect()

            print("Constructing validation graphs...")
            val_graphs = constructor.texts_to_graphs(X_val, y_val)
            del X_val
            gc.collect()

            print("Constructing test graphs...")
            test_graphs = constructor.texts_to_graphs(X_test, y_test)
            del X_test
            gc.collect()

        print("Graph construction complete!")

        trainer = GNNTrainer(self.config)
        trained_models = trainer.train_all_gnn_models(train_graphs, val_graphs)
        del train_graphs, val_graphs, test_graphs
        gc.collect()

        print(f"GNN training complete! Trained {len(trained_models)} models.")

    def run_hybrid_training(self):
        """Step 8: Train hybrid models"""
        self.print_step(8, 10, "TRAINING HYBRID ENSEMBLE MODELS")
        
        from models_hybrid import HybridModelTrainer
        
        feature_dir = f"{self.config['base_dir']}/features"
        data_dir = f"{self.config['base_dir']}/data"
        
        X_train = np.load(f"{feature_dir}/X_train_uniembed.npy")
        X_val = np.load(f"{feature_dir}/X_val_uniembed.npy")
        y_train = np.load(f"{data_dir}/y_train.npy")
        y_val = np.load(f"{data_dir}/y_val.npy")
        
        X_bert_train = X_bert_val = None
        bert_train_path = f"{feature_dir}/X_train_bert.npy"
        bert_val_path = f"{feature_dir}/X_val_bert.npy"
        
        if os.path.exists(bert_train_path) and os.path.exists(bert_val_path):
            X_bert_train = np.load(bert_train_path)
            X_bert_val = np.load(bert_val_path)
            print("Loaded BERT embeddings.")
        else:
            print("BERT embeddings not found, skipping BERT_XGBoost hybrid.")
        
        models_to_train = []
        for name in HYBRID_MODELS:
            if name == 'BERT_XGBoost' and X_bert_train is None:
                continue
            models_to_train.append(name)
        
        trainer = HybridModelTrainer(self.config)
        trained_models = trainer.train_all_hybrid_models(
            X_train, y_train, X_val, y_val,
            X_bert_train=X_bert_train, X_bert_val=X_bert_val,
            models_override=models_to_train
        )
        
        print(f"Hybrid training complete! Trained {len(trained_models)} models.")
    
    def run_evaluation(self):
        """Step 9: Evaluate all models — internal test + cross-dataset test"""
        self.print_step(9, 10, "EVALUATING ALL MODELS")

        from evaluation import ComprehensiveEvaluator

        feature_dir = f"{self.config['base_dir']}/features"
        data_dir    = f"{self.config['base_dir']}/data"

        X_test_text = np.load(f"{data_dir}/X_test.npy", allow_pickle=True)
        y_test      = np.load(f"{data_dir}/y_test.npy")

        test_data = {
            'X_test_tfidf':   np.load(f"{feature_dir}/X_test_tfidf.npy"),
            'X_test_uniembed': np.load(f"{feature_dir}/X_test_uniembed.npy"),
            'X_test_text':    X_test_text,
            'y_test':         y_test,
        }

        import torch
        from models_gnn import GraphConstructor, GNNTrainer
        from gensim.models import Word2Vec


        gnn_model_dir = f"{self.config['base_dir']}/models/gnn"
        if os.path.exists(f"{gnn_model_dir}/GCN.pt") or os.path.exists(f"{gnn_model_dir}/GAT.pt"):
            print("  Rebuilding test graphs for GNN evaluation...")
            try:
                word2vec_model = Word2Vec.load(f"{feature_dir}/word2vec.model")
                constructor = GraphConstructor(word2vec_model)
                test_data['test_graphs'] = constructor.texts_to_graphs(X_test_text, y_test)
                print(f"  Built {len(test_data['test_graphs'])} test graphs.")
                del word2vec_model
            except Exception as e:
                print(f"  WARNING: Could not build test graphs: {e}")

        evaluator   = ComprehensiveEvaluator(self.config)
        all_results = evaluator.evaluate_all_models(test_data)


        cross_text_path  = f"{data_dir}/X_cross.npy"
        cross_label_path = f"{data_dir}/y_cross.npy"

        if os.path.exists(cross_text_path) and os.path.exists(cross_label_path):
            print("\n" + "-"*70)
            print("CROSS-DATASET EVALUATION (Modified_SQL_Dataset — never seen in training)")
            print("-"*70)

            X_cross_text = np.load(cross_text_path, allow_pickle=True)
            y_cross      = np.load(cross_label_path)

            cross_tfidf_path  = f"{feature_dir}/X_cross_tfidf.npy"
            cross_embed_path  = f"{feature_dir}/X_cross_uniembed.npy"

            if os.path.exists(cross_tfidf_path) and os.path.exists(cross_embed_path):
                cross_data = {
                    'X_test_tfidf':    np.load(cross_tfidf_path),
                    'X_test_uniembed': np.load(cross_embed_path),
                    'X_test_text':     X_cross_text,
                    'y_test':          y_cross,
                }
                # cross_results saved with different filename so it doesn't
                # overwrite internal test results
                cross_results = evaluator.evaluate_all_models(cross_data)
                # Save cross results separately
                import pandas as pd
                cross_df = pd.DataFrame(cross_results).T if isinstance(cross_results, dict) else pd.DataFrame(cross_results)
                cross_df.to_csv(f"{self.config['results_dir']}/metrics/cross_dataset_metrics.csv", index=True)
                print(f"Cross-dataset evaluation complete! Results saved to cross_dataset_metrics.csv")
            else:
                print("  WARNING: Cross-dataset features not found.")
        else:
            print("\n  NOTE: No cross-dataset split found (X_cross.npy missing).")

        print(f"\nEvaluation complete! Evaluated {len(all_results)} models.")
        return all_results
    
    def run_visualization(self):
        """Step 10: Generate visualizations"""
        self.print_step(10, 10, "GENERATING VISUALIZATIONS")
        
        from visualization import VisualizationGenerator
        import pandas as pd
        
        results_path = f"{self.config['results_dir']}/metrics/all_models_metrics.csv"
        results_df = pd.read_csv(results_path)
        
        viz_gen = VisualizationGenerator(self.config)
        viz_gen.generate_all_visualizations(results_df)
        
        print("Visualization complete!")
    
    def run_full_pipeline(self, skip_steps=None):
        """Run complete pipeline"""
        if skip_steps is None:
            skip_steps = []
        
        print("="*70)
        print("    COMPREHENSIVE WEB ATTACK DETECTION RESEARCH PIPELINE    ")
        print("="*70)
        print(f"\nStart: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Output: {self.config['base_dir']}\n")
        
        try:
            if 1 not in skip_steps:
                self.run_setup()
            
            if 2 not in skip_steps:
                splits = self.run_data_preprocessing()
            else:
                splits = None
            
            if 3 not in skip_steps:
                self.run_feature_extraction(splits if 2 not in skip_steps else None)
            
            if 4 not in skip_steps and self.config.get('train_classical_ml', True):
                self.run_classical_ml_training()
            
            if 5 not in skip_steps and self.config.get('train_deep_learning', True):
                self.run_deep_learning_training()
            
            if 6 not in skip_steps and self.config.get('train_transformers', False):
                self.run_transformer_training()

            # Hybrid runs BEFORE GNN because GNN deletes feature files from disk
            # during its disk-cleanup phase. Hybrid needs X_train_uniembed.npy etc.
            if 8 not in skip_steps and self.config.get('train_hybrid', True) and not self.config.get('skip_hybrid', False):
                self.run_hybrid_training()

            if 7 not in skip_steps and self.config.get('train_gnn', False):
                self.run_gnn_training()
            
            if 9 not in skip_steps:
                all_results = self.run_evaluation()
            
            if 10 not in skip_steps:
                self.run_visualization()
            
            print("\n" + "="*70)
            print("    PIPELINE COMPLETE!    ")
            print("="*70)
            print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except Exception as e:
            print("\n" + "="*70)
            print("ERROR: Pipeline failed!")
            print("="*70)
            print(f"Error: {str(e)}\n")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--skip-steps', nargs='+', type=int, default=[],
                        help='Steps to skip (1-10)')
    args = parser.parse_args()
    
    pipeline = MasterPipeline(GLOBAL_CONFIG)
    pipeline.run_full_pipeline(skip_steps=args.skip_steps)

