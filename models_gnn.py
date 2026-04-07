
from config import GLOBAL_CONFIG, GNN_MODELS

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv, global_mean_pool
import networkx as nx
from pathlib import Path
import time
import json
import gc


class GraphConstructor:
    """Convert text sequences to graph representations"""
    
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        
    def _get_node_type(self, token):
        """Get node type one-hot encoding"""
        sql_keywords = {'select', 'insert', 'update', 'delete', 'union', 'where', 'from', 'and', 'or'}
        xss_keywords = {'script', 'alert', 'onerror', 'onload', 'img', 'iframe'}
        
        if token.lower() in sql_keywords:
            return np.array([1, 0, 0, 0])  # SQL keyword
        elif token.lower() in xss_keywords:
            return np.array([0, 1, 0, 0])  # XSS keyword
        elif token.startswith('<') or token.startswith('>'):
            return np.array([0, 0, 1, 0])  # Syntax token
        else:
            return np.array([0, 0, 0, 1])  # Normal token
    
    def _is_attack_token(self, token):
        """Heuristic to identify attack-related tokens"""
        attack_patterns = ['<', '>', 'script', 'select', 'union', 'alert', 'or', 'and', '=', '--']
        return any(pattern in token.lower() for pattern in attack_patterns)
    
    def text_to_graph(self, text, label):
        """Convert text to graph with FIXED dimensions"""
        tokens = text.split()
        
        if len(tokens) == 0:
            tokens = ['<EMPTY>']
        
        # Get Word2Vec dimension 
        if self.word2vec_model and len(self.word2vec_model.wv) > 0:
            sample_word = list(self.word2vec_model.wv.index_to_key)[0]
            w2v_dim = len(self.word2vec_model.wv[sample_word])
        else:
            w2v_dim = 50
        
        # Total dimension: w2v_dim + 4 (type) + 1 (pos) + 1 (attack) + 1 (freq) = w2v_dim + 7
        total_dim = w2v_dim + 7
        
        node_features = []
        for i, token in enumerate(tokens):
            # Token embedding
            if self.word2vec_model and token in self.word2vec_model.wv:
                token_emb = self.word2vec_model.wv[token]
            else:
                token_emb = np.zeros(w2v_dim)
            
            # Node type (4D one-hot)
            node_type = self._get_node_type(token)
            
            # Positional encoding
            pos_encoding = i / len(tokens)
            
            # Attack indicator
            attack_indicator = 1.0 if self._is_attack_token(token) else 0.0
            
            # Token frequency
            token_freq = tokens.count(token) / len(tokens)
            
            # Combine features
            node_feat = np.concatenate([
                token_emb,           # w2v_dim
                node_type,           # 4
                [pos_encoding],      # 1
                [attack_indicator],  # 1
                [token_freq]         # 1
            ])
            
            assert len(node_feat) == total_dim, f"Dimension mismatch: {len(node_feat)} vs {total_dim}"
            node_features.append(node_feat)
        
        edge_index = []
        edge_attr = []
        

        for i in range(len(tokens) - 1):
            edge_index.append([i, i + 1])
            edge_attr.append([1.0, 0.0, 1.0])
            edge_index.append([i + 1, i])
            edge_attr.append([1.0, 0.0, 1.0])
        
        if self.word2vec_model:
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    if tokens[i] in self.word2vec_model.wv and tokens[j] in self.word2vec_model.wv:
                        try:
                            similarity = self.word2vec_model.wv.similarity(tokens[i], tokens[j])
                            if similarity > 0.7:
                                edge_index.append([i, j])
                                edge_attr.append([0.0, 1.0, similarity])
                                edge_index.append([j, i])
                                edge_attr.append([0.0, 1.0, similarity])
                        except:
                            pass
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    
    def texts_to_graphs(self, texts, labels):
        """Convert multiple texts to graphs"""
        print("Converting texts to graphs...")
        graphs = []
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            graph = self.text_to_graph(text, label)
            graphs.append(graph)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(texts)} graphs")
        
        print(f"Created {len(graphs)} graphs")
        return graphs


class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.fc = nn.Linear(hidden_dims[2], num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, input_dim, hidden_dims, num_classes, heads=4, dropout=0.3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dims[0], heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dims[1] * heads, hidden_dims[2], heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dims[2], num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GNNTrainer:
    """Train and evaluate GNN models with proper device handling"""
    
    def __init__(self, config=None):
        if config is None:
            config = GLOBAL_CONFIG
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_model(self, model_name, input_dim):
        """Create GNN model by name"""
        print(f"Creating {model_name} with input_dim={input_dim}")
        
        models = {
            'GCN': GCN(
                input_dim=input_dim,
                hidden_dims=self.config['gnn_hidden_dims'],
                num_classes=2,
                dropout=self.config['gnn_dropout']
            ),
            'GAT': GAT(
                input_dim=input_dim,
                hidden_dims=self.config['gnn_hidden_dims'],
                num_classes=2,
                heads=self.config['gat_heads'],
                dropout=self.config['gnn_dropout']
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown GNN model: {model_name}")
        
        return models[model_name]
    
    def train_epoch(self, model, loader, optimizer):
        """Train for one epoch - DEVICE FIXED"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item() * batch.num_graphs
        
        return total_loss / len(loader.dataset), correct / total
    
    def evaluate(self, model, loader):
        """Evaluate model - DEVICE FIXED"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return correct / total
    
    def train_model(self, model_name, train_graphs, val_graphs):
        """Train a single GNN model"""
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}\n")
        
        input_dim = train_graphs[0].x.shape[1]
        print(f"   Train graphs: {len(train_graphs)}")
        print(f"   Val graphs: {len(val_graphs)}")
        print(f"   Node feature dim: {input_dim}")
        
        model = self.create_model(model_name, input_dim)
        model = model.to(self.device)  
        
        print(f"\nModel: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
        

        train_loader = DataLoader(train_graphs, batch_size=self.config['gnn_batch_size'], shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=self.config['gnn_batch_size'], shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['gnn_lr'])
        
        if model_name == 'GAT' and 'gat_epochs' in self.config:
            num_epochs = self.config['gat_epochs']
        else:
            num_epochs = self.config['gnn_epochs']

        print(f"\nTraining for {num_epochs} epochs...")
        best_val_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer)
            val_acc = self.evaluate(model, val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        print(f"\nTraining complete in {training_time:.2f} seconds")
        
        model_info = {
            'model_name': model_name,
            'training_time': training_time,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'best_val_accuracy': best_val_acc,
            'input_dim': input_dim
        }
        
        return model, history, model_info
    
    def save_model(self, model, model_name, model_info, history, save_dir=None):
        """Save trained GNN model"""
        if save_dir is None:
            save_dir = f"{self.config['models_dir']}/gnn"
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        

        model_path = f"{save_dir}/{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
        

        metadata_path = f"{save_dir}/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
        
        history_path = f"{save_dir}/{model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"History saved: {history_path}")
    
    def train_all_gnn_models(self, train_graphs, val_graphs):
        """Train all GNN models — skips any model whose .pt file already exists."""
        print(f"\n{'='*70}")
        print(f"TRAINING ALL GNN MODELS")
        print(f"{'='*70}\n")

        save_dir = f"{self.config['models_dir']}/gnn"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        trained_models = {}
        models_to_train = ['GCN', 'GAT']

        for model_name in models_to_train:
            model_path = f"{save_dir}/{model_name}.pt"

            if os.path.exists(model_path):
                print(f"{'='*70}")
                print(f"SKIPPING: {model_name} (already trained — {model_path})")
                print(f"{'='*70}\n")
                trained_models[model_name] = {'skipped': True}
                continue

            try:
                model, history, model_info = self.train_model(
                    model_name, train_graphs, val_graphs
                )

                self.save_model(model, model_name, model_info, history)

                trained_models[model_name] = {
                    'model': model,
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
                continue

        print(f"\n{'='*70}")
        print(f" ALL GNN MODELS TRAINED: {len(trained_models)}/{len(models_to_train)}")
        print(f"{'='*70}\n")

        return trained_models


if __name__ == "__main__":
    from config import GLOBAL_CONFIG, set_seed
    import pickle
    
    set_seed(GLOBAL_CONFIG['random_seed'])
    
    data_dir = f"{GLOBAL_CONFIG['base_dir']}/data"
    
    with open(f"{data_dir}/train_graphs.pkl", 'rb') as f:
        train_graphs = pickle.load(f)
    
    with open(f"{data_dir}/val_graphs.pkl", 'rb') as f:
        val_graphs = pickle.load(f)
    
    trainer = GNNTrainer(GLOBAL_CONFIG)
    trained_models = trainer.train_all_gnn_models(train_graphs, val_graphs)
    
    print("GNN training complete!")

