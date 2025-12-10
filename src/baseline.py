import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import negative_sampling
try:
    from torch_geometric.nn.models import MIPSKNNIndex
except ImportError:
    MIPSKNNIndex = None

try:
    from torch_geometric.metrics import LinkPredMAP, LinkPredPrecision, LinkPredRecall
except ImportError:
    LinkPredMAP = None
    LinkPredPrecision = None
    LinkPredRecall = None

from sklearn.metrics import roc_auc_score
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Data paths
TRAIN_DATA_PATH = '/Users/hnasrolahi/Desktop/cs782project/train_data.pt'
VAL_DATA_PATH = '/Users/hnasrolahi/Desktop/cs782project/val_data.pt'
TEST_DATA_PATH = '/Users/hnasrolahi/Desktop/cs782project/test_data.pt'

# Model hyperparameters
HIDDEN_CHANNELS = 128
LEARNING_RATE = 0.001
EPOCHS = 100

# Train both loss functions
TRAIN_BOTH = True  # Set to True to train both BCE and BPR

# Evaluation K values
K_VALUES = [5, 20, 100]

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed
RANDOM_SEED = 42
# =====================================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Fallback MIPS implementation
if MIPSKNNIndex is None:
    class SimpleMIPSKNNIndex:
        def __init__(self, embeddings):
            self.embeddings = embeddings
        
        def search(self, query_embeddings, k):
            scores = torch.matmul(query_embeddings, self.embeddings.t())
            top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
            return top_k_scores, top_k_indices
    
    MIPSKNNIndex = SimpleMIPSKNNIndex


# Fallback metric implementations
if LinkPredPrecision is None or LinkPredRecall is None:
    
    class SimpleLinkPredPrecision:
        def __init__(self, k):
            self.k = k
            self.total_precision = 0.0
            self.num_samples = 0
        
        def to(self, device):
            return self
        
        def update(self, pred_index_mat, edge_label_index):
            ground_truth = {}
            for i in range(edge_label_index.shape[1]):
                user = edge_label_index[0, i].item()
                repo = edge_label_index[1, i].item()
                if user not in ground_truth:
                    ground_truth[user] = set()
                ground_truth[user].add(repo)
            
            for user_idx in range(pred_index_mat.shape[0]):
                if user_idx not in ground_truth:
                    continue
                
                predicted_repos = pred_index_mat[user_idx].cpu().numpy()
                true_repos = ground_truth[user_idx]
                
                hits = sum(1 for repo in predicted_repos if repo in true_repos)
                precision = hits / self.k
                
                self.total_precision += precision
                self.num_samples += 1
        
        def compute(self):
            if self.num_samples == 0:
                return 0.0
            return self.total_precision / self.num_samples
    
    class SimpleLinkPredRecall:
        def __init__(self, k):
            self.k = k
            self.total_recall = 0.0
            self.num_samples = 0
        
        def to(self, device):
            return self
        
        def update(self, pred_index_mat, edge_label_index):
            ground_truth = {}
            for i in range(edge_label_index.shape[1]):
                user = edge_label_index[0, i].item()
                repo = edge_label_index[1, i].item()
                if user not in ground_truth:
                    ground_truth[user] = set()
                ground_truth[user].add(repo)
            
            for user_idx in range(pred_index_mat.shape[0]):
                if user_idx not in ground_truth:
                    continue
                
                predicted_repos = pred_index_mat[user_idx].cpu().numpy()
                true_repos = ground_truth[user_idx]
                
                hits = sum(1 for repo in predicted_repos if repo in true_repos)
                recall = hits / len(true_repos) if len(true_repos) > 0 else 0.0
                
                self.total_recall += recall
                self.num_samples += 1
        
        def compute(self):
            if self.num_samples == 0:
                return 0.0
            return self.total_recall / self.num_samples
    
    LinkPredPrecision = SimpleLinkPredPrecision
    LinkPredRecall = SimpleLinkPredRecall


class GNNEncoder(torch.nn.Module):
    """GraphSAGE-based encoder"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    """Edge decoder for link prediction"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.user_lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.repo_lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        user_emb = z_dict['user'][row]
        repo_emb = z_dict['repo'][col]
        return (self.user_lin(user_emb) * self.repo_lin(repo_emb)).sum(dim=-1)


class Model(torch.nn.Module):
    """Complete model"""
    def __init__(self, hidden_channels, metadata, device=None):
        super().__init__()
        self.device = device or DEVICE

        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        
        # Add reverse edges to metadata
        node_types, edge_types = metadata
        reverse_edge = ('repo', 'rev_contributes_to', 'user')
        if reverse_edge not in edge_types:
            edge_types = edge_types + [reverse_edge]
        metadata = (node_types, edge_types)
        
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)
        self.to(self.device)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def add_reverse_edges(data):
    """Add reverse edges for bipartite graph"""
    edge_index = data['user', 'contributes_to', 'repo'].edge_index
    reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    data['repo', 'rev_contributes_to', 'user'].edge_index = reverse_edge_index
    return data


def bce_loss(predicted, actual):
    """Binary Cross-Entropy loss"""
    return F.binary_cross_entropy_with_logits(predicted, actual)


def bpr_loss(pos_pred, neg_pred):
    """Bayesian Personalized Ranking loss"""
    return -torch.log(torch.sigmoid(pos_pred - neg_pred) + 1e-10).mean()


def train_epoch_bce(model, train_data, optimizer):
    """Train one epoch with BCE loss"""
    model.train()
    optimizer.zero_grad()

    # Get positive edges
    pos_edge_label_index = train_data['user', 'contributes_to', 'repo'].edge_index
    
    # Generate negative edges
    neg_edge_label_index = negative_sampling(
        edge_index=pos_edge_label_index,
        num_nodes=(train_data['user'].x.size(0), train_data['repo'].x.size(0)),
        num_neg_samples=pos_edge_label_index.size(1),
    )
    
    # Combine positive and negative edges
    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_label_index.size(1)),
        torch.zeros(neg_edge_label_index.size(1))
    ]).to(DEVICE)
    
    # Forward pass
    predicted = model(train_data.x_dict, train_data.edge_index_dict, edge_label_index)
    
    # Compute loss
    loss = bce_loss(predicted, edge_label)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return float(loss)


def train_epoch_bpr(model, train_data, optimizer):
    """Train one epoch with BPR loss"""
    model.train()
    optimizer.zero_grad()

    # Get positive edges
    pos_edge_index = train_data['user', 'contributes_to', 'repo'].edge_index
    
    # Generate negative edges (same number as positive)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(train_data['user'].x.size(0), train_data['repo'].x.size(0)),
        num_neg_samples=pos_edge_index.size(1),
    )
    
    # Forward pass for positive and negative samples
    pos_pred = model(train_data.x_dict, train_data.edge_index_dict, pos_edge_index)
    neg_pred = model(train_data.x_dict, train_data.edge_index_dict, neg_edge_index)
    
    # Compute BPR loss
    loss = bpr_loss(pos_pred, neg_pred)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return float(loss)


@torch.no_grad()
def evaluate_auc(model, data):
    """Evaluate AUC-ROC"""
    model.eval()
    
    pos_edge_label_index = data['user', 'contributes_to', 'repo'].edge_index
    neg_edge_label_index = negative_sampling(
        edge_index=pos_edge_label_index,
        num_nodes=(data['user'].x.size(0), data['repo'].x.size(0)),
        num_neg_samples=pos_edge_label_index.size(1),
    )
    
    edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_label_index.size(1)),
        torch.zeros(neg_edge_label_index.size(1))
    ]).cpu().numpy()
    
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    pred = torch.sigmoid(pred).cpu().numpy()
    
    auc = roc_auc_score(edge_label, pred)
    return auc


@torch.no_grad()
def evaluate_retrieval(model, data, k=10):
    """Evaluate Precision@K and Recall@K"""
    model.eval()
    
    # Compute node embeddings
    emb = model.encoder(data.x_dict, data.edge_index_dict)
    repo_emb = emb['repo']
    user_emb = emb['user']
    
    # Apply decoder transformations
    repo_embeddings = model.decoder.repo_lin(repo_emb)
    user_embeddings = model.decoder.user_lin(user_emb)
    
    # Get edge label indices
    edge_label_index = data['user', 'contributes_to', 'repo'].edge_index
    
    # MIPS search
    mips = MIPSKNNIndex(repo_embeddings)
    _, pred_index_mat = mips.search(user_embeddings, k)
    
    # Initialize metrics
    precision_metric = LinkPredPrecision(k=k).to(DEVICE)
    recall_metric = LinkPredRecall(k=k).to(DEVICE)
    
    # Update metrics
    precision_metric.update(pred_index_mat, edge_label_index)
    recall_metric.update(pred_index_mat, edge_label_index)
    
    return float(precision_metric.compute()), float(recall_metric.compute())


def plot_combined_metrics(results_bce, results_bpr, k_values):
    """Generate plots comparing BCE and BPR for all K values"""
    
    # Unpack common results
    train_losses_bce = results_bce['train_losses']
    val_aucs_bce = results_bce['val_aucs']
    epochs_list_bce = results_bce['epochs_list']
    best_epoch_bce = results_bce['best_epoch']
    
    train_losses_bpr = results_bpr['train_losses']
    val_aucs_bpr = results_bpr['val_aucs']
    epochs_list_bpr = results_bpr['epochs_list']
    best_epoch_bpr = results_bpr['best_epoch']
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses_bce) + 1), train_losses_bce, 'b-', 
             linewidth=2, label='BCE Loss', alpha=0.8)
    plt.plot(range(1, len(train_losses_bpr) + 1), train_losses_bpr, 'r-', 
             linewidth=2, label='BPR Loss', alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison: BCE vs BPR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/hnasrolahi/Desktop/cs782project/training_loss_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training loss comparison plot saved")
    
    # 2. Validation AUC Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list_bce, val_aucs_bce, 'b-o', linewidth=2, markersize=6, 
             label='BCE', alpha=0.8)
    plt.plot(epochs_list_bpr, val_aucs_bpr, 'r-s', linewidth=2, markersize=6, 
             label='BPR', alpha=0.8)
    plt.axvline(x=best_epoch_bce, color='b', linestyle='--', linewidth=1.5, 
                alpha=0.5, label=f'Best BCE (epoch {best_epoch_bce})')
    plt.axvline(x=best_epoch_bpr, color='r', linestyle='--', linewidth=1.5, 
                alpha=0.5, label=f'Best BPR (epoch {best_epoch_bpr})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation AUC', fontsize=12)
    plt.title('Validation AUC Comparison: BCE vs BPR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/hnasrolahi/Desktop/cs782project/val_auc_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Validation AUC comparison plot saved")
    
    # 3-N. Generate Recall@K and Precision@K plots for each K value
    for k in k_values:
        recall_key = f'val_recall{k}'
        precision_key = f'val_precision{k}'
        
        # Validation Recall@K Comparison
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_list_bce, results_bce[recall_key], 'b-^', linewidth=2, markersize=6, 
                 label='BCE', alpha=0.8)
        plt.plot(epochs_list_bpr, results_bpr[recall_key], 'r-^', linewidth=2, markersize=6, 
                 label='BPR', alpha=0.8)
        plt.axvline(x=best_epoch_bce, color='b', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axvline(x=best_epoch_bpr, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'Recall@{k}', fontsize=12)
        plt.title(f'Validation Recall@{k} Comparison: BCE vs BPR', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'/Users/hnasrolahi/Desktop/cs782project/val_recall{k}_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Validation Recall@{k} comparison plot saved")
        
        # Validation Precision@K Comparison
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_list_bce, results_bce[precision_key], 'b-s', linewidth=2, markersize=6, 
                 label='BCE', alpha=0.8)
        plt.plot(epochs_list_bpr, results_bpr[precision_key], 'r-s', linewidth=2, markersize=6, 
                 label='BPR', alpha=0.8)
        plt.axvline(x=best_epoch_bce, color='b', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axvline(x=best_epoch_bpr, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'Precision@{k}', fontsize=12)
        plt.title(f'Validation Precision@{k} Comparison: BCE vs BPR', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'/Users/hnasrolahi/Desktop/cs782project/val_precision{k}_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Validation Precision@{k} comparison plot saved")


def train_model_with_loss(loss_name, train_data, val_data, test_data, k_values):
    """Train model with specified loss function"""
    print(f"\n{'='*80}")
    print(f"TRAINING WITH {loss_name.upper()} LOSS")
    print('='*80)
    
    # Initialize model
    model = Model(HIDDEN_CHANNELS, train_data.metadata(), DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize lazy parameters
    with torch.no_grad():
        dummy_edge_index = train_data['user', 'contributes_to', 'repo'].edge_index[:, :10]
        _ = model(train_data.x_dict, train_data.edge_index_dict, dummy_edge_index)
    
    print(f"✓ Model initialized ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Select training function
    train_fn = train_epoch_bce if loss_name == 'bce' else train_epoch_bpr
    
    # Tracking
    results = {
        'train_losses': [],
        'val_aucs': [],
        'epochs_list': [],
        'best_epoch': 0,
        'best_val_auc': 0
    }
    
    # Add tracking for each K value
    for k in k_values:
        results[f'val_recall{k}'] = []
        results[f'val_precision{k}'] = []
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        loss = train_fn(model, train_data, optimizer)
        results['train_losses'].append(loss)
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            val_auc = evaluate_auc(model, val_data)
            results['val_aucs'].append(val_auc)
            results['epochs_list'].append(epoch)
            
            # Evaluate for all K values
            for k in k_values:
                prec_k, rec_k = evaluate_retrieval(model, val_data, k=k)
                results[f'val_recall{k}'].append(rec_k)
                results[f'val_precision{k}'].append(prec_k)
            
            if epoch % 10 == 0:  # Print every 10 epochs
                metrics_str = " | ".join([f"R@{k}: {results[f'val_recall{k}'][-1]:.4f}" for k in k_values])
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {val_auc:.4f} | {metrics_str}")
            
            if val_auc > results['best_val_auc']:
                results['best_val_auc'] = val_auc
                results['best_epoch'] = epoch
                torch.save(model.state_dict(), 
                          f'/Users/hnasrolahi/Desktop/cs782project/best_model_{loss_name}.pt')
    
    print(f"\n✓ Training complete | Best AUC: {results['best_val_auc']:.4f} (epoch {results['best_epoch']})")
    
    # Load best model
    model.load_state_dict(torch.load(
        f'/Users/hnasrolahi/Desktop/cs782project/best_model_{loss_name}.pt', 
        weights_only=True))
    
    # Test evaluation
    results['test_auc'] = evaluate_auc(model, test_data)
    
    print(f"\nTest Results:")
    print(f"  AUC: {results['test_auc']:.4f}")
    
    for k in k_values:
        test_prec_k, test_rec_k = evaluate_retrieval(model, test_data, k=k)
        results[f'test_prec{k}'] = test_prec_k
        results[f'test_rec{k}'] = test_rec_k
        print(f"  Precision@{k}: {test_prec_k:.4f} | Recall@{k}: {test_rec_k:.4f}")
    
    return results


def main():
    print("="*80)
    print("BASELINE GNN MODEL - USER-REPO LINK PREDICTION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Training with: BCE and BPR loss functions")
    print(f"K values for evaluation: {K_VALUES}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_data = torch.load(TRAIN_DATA_PATH, weights_only=False)
    train_data = add_reverse_edges(train_data).to(DEVICE)
    
    val_data = torch.load(VAL_DATA_PATH, weights_only=False)
    val_data = add_reverse_edges(val_data).to(DEVICE)
    
    test_data = torch.load(TEST_DATA_PATH, weights_only=False)
    test_data = add_reverse_edges(test_data).to(DEVICE)
    
    print(f"✓ Data loaded")
    print(f"  Users: {train_data['user'].x.size(0)}")
    print(f"  Repos: {train_data['repo'].x.size(0)}")
    print(f"  Training edges: {train_data['user', 'contributes_to', 'repo'].edge_index.size(1)}")
    
    # Train with BCE
    results_bce = train_model_with_loss('bce', train_data, val_data, test_data, K_VALUES)
    
    # Train with BPR
    results_bpr = train_model_with_loss('bpr', train_data, val_data, test_data, K_VALUES)
    
    # Generate comparison plots
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    
    plot_combined_metrics(results_bce, results_bpr, K_VALUES)
    
    print("\n✓ All comparison plots generated")
    
    # Save combined results
    results_file = '/Users/hnasrolahi/Desktop/cs782project/results_comparison.txt'
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODEL RESULTS - BCE vs BPR COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("BCE Loss Results:\n")
        f.write(f"  Best epoch: {results_bce['best_epoch']}\n")
        f.write(f"  Best validation AUC: {results_bce['best_val_auc']:.4f}\n")
        f.write(f"  Test AUC: {results_bce['test_auc']:.4f}\n")
        for k in K_VALUES:
            f.write(f"  Test Precision@{k}: {results_bce[f'test_prec{k}']:.4f} | Recall@{k}: {results_bce[f'test_rec{k}']:.4f}\n")
        
        f.write("\nBPR Loss Results:\n")
        f.write(f"  Best epoch: {results_bpr['best_epoch']}\n")
        f.write(f"  Best validation AUC: {results_bpr['best_val_auc']:.4f}\n")
        f.write(f"  Test AUC: {results_bpr['test_auc']:.4f}\n")
        for k in K_VALUES:
            f.write(f"  Test Precision@{k}: {results_bpr[f'test_prec{k}']:.4f} | Recall@{k}: {results_bpr[f'test_rec{k}']:.4f}\n")
        
        f.write("\nWinner Analysis:\n")
        f.write(f"  Better Test AUC: {'BCE' if results_bce['test_auc'] > results_bpr['test_auc'] else 'BPR'}\n")
        for k in K_VALUES:
            winner = 'BCE' if results_bce[f'test_rec{k}'] > results_bpr[f'test_rec{k}'] else 'BPR'
            f.write(f"  Better Recall@{k}: {winner}\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBCE Test AUC: {results_bce['test_auc']:.4f} | BPR Test AUC: {results_bpr['test_auc']:.4f}")
    for k in K_VALUES:
        print(f"K={k}: BCE Recall@{k}: {results_bce[f'test_rec{k}']:.4f} | BPR Recall@{k}: {results_bpr[f'test_rec{k}']:.4f}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()