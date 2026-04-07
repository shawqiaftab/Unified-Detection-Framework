
from config import GLOBAL_CONFIG, TRANSFORMER_MODELS

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from tqdm.auto import tqdm
import time
import json
from pathlib import Path
import gc


class AttackDetectionDataset(Dataset):
    """Custom dataset for transformer models"""
    
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerModelFactory:
    """Factory for creating transformer models"""
    
    MODEL_CONFIGS = {
        'DistilBERT': {
            'model_name': 'distilbert-base-uncased',
            'tokenizer_class': DistilBertTokenizer,
            'model_class': DistilBertForSequenceClassification
        },
        'BERT': {
            'model_name': 'bert-base-uncased',
            'tokenizer_class': BertTokenizer,
            'model_class': BertForSequenceClassification
        }
    }
    
    def create_model_and_tokenizer(self, model_name):
        """Create transformer model and tokenizer"""
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.MODEL_CONFIGS[model_name]
        
        print(f"Loading {config['model_name']}...")
        
        tokenizer = config['tokenizer_class'].from_pretrained(config['model_name'])
        model = config['model_class'].from_pretrained(
            config['model_name'],
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        print(f"{model_name} loaded successfully")
        
        return model, tokenizer


class TransformerTrainer:
    """Train transformer models with extreme memory optimization"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.factory = TransformerModelFactory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val, tokenizer):
        """Create PyTorch data loaders"""
        
        train_dataset = AttackDetectionDataset(
            X_train, y_train, tokenizer, self.config['max_seq_length']
        )
        val_dataset = AttackDetectionDataset(
            X_val, y_val, tokenizer, self.config['max_seq_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['transformer_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False  
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['transformer_batch_size'] * 2,  
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, scaler):
        """Train for one epoch with gradient accumulation and mixed precision"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        accumulation_steps = self.config.get('gradient_accumulation_steps', 8)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision training
            if self.config.get('fp16', True):
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config.get('fp16', True):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get('max_grad_norm', 1.0))
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            logits = outputs.logits
            total_loss += loss.item() * accumulation_steps
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
     
            if (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
    
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, model, val_loader):
        """Evaluate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.get('fp16', True):
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Fine-tune a transformer model with all optimizations"""
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING: {model_name}")
        print(f"{'='*70}\n")
        
  
        if self.config.get('use_sampled_data_for_transformers', False):
            sample_fraction = self.config.get('transformer_sample_fraction', 0.1)
            sample_size = int(len(X_train) * sample_fraction)
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            print(f"Using {len(X_train)} samples ({sample_fraction*100}% of training data)")
        

        model, tokenizer = self.factory.create_model_and_tokenizer(model_name)
        model.to(self.device)
        

        print("Creating data loaders...")
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, tokenizer
        )
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Effective batch size: {self.config['transformer_batch_size'] * self.config.get('gradient_accumulation_steps', 8)}")
        

        optimizer = AdamW(
            model.parameters(),
            lr=self.config['transformer_lr'],
            eps=1e-8,
            weight_decay=0.01
        )
        

        total_steps = len(train_loader) * self.config['transformer_epochs']
        total_steps = total_steps // self.config.get('gradient_accumulation_steps', 8)
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.get('fp16', True))

        print(f"\nFine-tuning {model_name} for {self.config['transformer_epochs']} epochs...")
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config['transformer_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['transformer_epochs']}")
            
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, scheduler, scaler)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            val_loss, val_acc = self.evaluate(model, val_loader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"   New best model!")
            
            torch.cuda.empty_cache()
            gc.collect()
        
        training_time = time.time() - start_time
        
        print(f"\nFine-tuning complete in {training_time:.2f} seconds")
        
        # Model info
        model_info = {
            'model_name': model_name,
            'training_time': training_time,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'epochs_trained': self.config['transformer_epochs'],
            'best_val_loss': float(best_val_loss),
            'best_val_accuracy': float(np.max(history['val_accuracy'])),
            'samples_used': len(X_train)
        }
        
        return model, tokenizer, history, model_info
    
    def save_model(self, model, tokenizer, model_name, model_info, history, save_dir=None):
        """Save fine-tuned model"""
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/transformers/{model_name}"
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved: {save_dir}")
        
        metadata_path = f"{save_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
        
        history_path = f"{save_dir}/history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"History saved: {history_path}")
    
    def train_all_transformer_models(self, X_train, y_train, X_val, y_val):
        """Train all transformer models (just DistilBERT for Colab)"""
        
        print(f"\n{'='*70}")
        print(f"FINE-TUNING ALL TRANSFORMER MODELS")
        print(f"{'='*70}\n")
        
        trained_models = {}
        
        models_to_train = ['DistilBERT']
        
        for model_name in models_to_train:
            try:
                model, tokenizer, history, model_info = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                self.save_model(model, tokenizer, model_name, model_info, history)
                
                trained_models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'history': history,
                    'info': model_info
                }
                
                print(f"{model_name} complete!\n")
                

                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}\n")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        print(f"\n{'='*70}")
        print(f"ALL TRANSFORMER MODELS TRAINED: {len(trained_models)}/{len(models_to_train)}")
        print(f"{'='*70}\n")
        
        return trained_models


if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    # Load raw text data
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    X_train = np.load(f"{data_dir}/X_train.npy", allow_pickle=True)
    X_val = np.load(f"{data_dir}/X_val.npy", allow_pickle=True)
    
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_val = np.load(f"{data_dir}/y_val.npy")
    
    # Train all models
    trainer = TransformerTrainer(GLOBAL_CONFIG)
    trained_models = trainer.train_all_transformer_models(
        X_train, y_train, X_val, y_val
    )
    
    print("Transformer fine-tuning complete!")

