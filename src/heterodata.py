"""
Create PyTorch Geometric HeteroData structure from feature parquet files
Includes node mappings for future evaluation
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import pickle
import numpy as np

# =========================== CONFIGURATION ===========================
# Input files
USER_FEATURES_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/user_features.parquet'
REPO_FEATURES_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/repo_features.parquet'
TRANSFORMED_EDGES_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/filtered_edges_10plus_transformed.parquet'

# Output files
HETERO_DATA_OUTPUT = '/Users/hnasrolahi/Desktop/cs782project/hetero_graph.pt'
NODE_MAPPINGS_OUTPUT = '/Users/hnasrolahi/Desktop/cs782project/node_mappings.pkl'
GRAPH_INFO_OUTPUT = '/Users/hnasrolahi/Desktop/cs782project/graph_info.txt'
# =====================================================================

def load_features(user_features_path, repo_features_path):
    """Load user and repo features from parquet files"""
    print("Loading node features...")
    
    user_df = pd.read_parquet(user_features_path)
    repo_df = pd.read_parquet(repo_features_path)
    
    print(f"  Users: {len(user_df)} nodes with {len(user_df.columns)-1} features")
    print(f"  Repos: {len(repo_df)} nodes with {len(repo_df.columns)-1} features")
    
    return user_df, repo_df


def create_node_mappings(user_df, repo_df):
    """Create bidirectional mappings between node names and indices"""
    print("\nCreating node mappings...")
    
    # Get node names
    users = user_df['user'].tolist()
    repos = repo_df['repo'].tolist()
    
    # Create mappings
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    
    repo_to_idx = {repo: idx for idx, repo in enumerate(repos)}
    idx_to_repo = {idx: repo for repo, idx in repo_to_idx.items()}
    
    print(f"  Created mappings for {len(users)} users and {len(repos)} repos")
    
    return {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'repo_to_idx': repo_to_idx,
        'idx_to_repo': idx_to_repo
    }


def build_hetero_data(user_df, repo_df, edges_df, mappings):
    """Build PyTorch Geometric HeteroData object"""
    print("\nBuilding HeteroData object...")
    
    data = HeteroData()
    
    # Add user node features
    print("  Adding user node features...")
    user_feature_cols = [col for col in user_df.columns if col != 'user']
    user_features = user_df[user_feature_cols].values.astype(np.float32)
    data['user'].x = torch.tensor(user_features, dtype=torch.float)
    print(f"    Shape: {data['user'].x.shape}")
    
    # Add repo node features
    print("  Adding repo node features...")
    repo_feature_cols = [col for col in repo_df.columns if col != 'repo']
    repo_features = repo_df[repo_feature_cols].values.astype(np.float32)
    data['repo'].x = torch.tensor(repo_features, dtype=torch.float)
    print(f"    Shape: {data['repo'].x.shape}")
    
    # Add edges
    print("  Adding edges...")
    user_indices = []
    repo_indices = []
    edge_weights = []
    
    skipped = 0
    for _, row in edges_df.iterrows():
        user_idx = mappings['user_to_idx'].get(row['user'])
        repo_idx = mappings['repo_to_idx'].get(row['repo'])
        
        if user_idx is not None and repo_idx is not None:
            user_indices.append(user_idx)
            repo_indices.append(repo_idx)
            edge_weights.append(row['log_weight'])
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"    Warning: Skipped {skipped} edges (nodes not in feature files)")
    
    # Create edge index tensor [2, num_edges]
    edge_index = torch.tensor([user_indices, repo_indices], dtype=torch.long)
    data['user', 'contributes_to', 'repo'].edge_index = edge_index
    print(f"    Edge index shape: {edge_index.shape}")
    
    # Add edge attributes (log_weight only)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    data['user', 'contributes_to', 'repo'].edge_attr = edge_attr
    print(f"    Edge attribute shape: {edge_attr.shape}")
    
    # Validate the graph
    print("\n  Validating graph...")
    data.validate()
    print("    ✓ Graph structure is valid")
    
    return data


def save_graph_info(data, mappings, output_path):
    """Save human-readable information about the graph"""
    print("\nSaving graph information...")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HETEROGENEOUS GRAPH INFORMATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Node Types:\n")
        f.write(f"  - Users: {len(mappings['user_to_idx'])} nodes\n")
        f.write(f"  - Repos: {len(mappings['repo_to_idx'])} nodes\n")
        f.write(f"  - Total: {len(mappings['user_to_idx']) + len(mappings['repo_to_idx'])} nodes\n\n")
        
        f.write("Node Features:\n")
        f.write(f"  - User feature dimensions: {data['user'].x.shape[1]}\n")
        f.write(f"  - Repo feature dimensions: {data['repo'].x.shape[1]}\n\n")
        
        f.write("Edge Type:\n")
        f.write(f"  - ('user', 'contributes_to', 'repo'): {data['user', 'contributes_to', 'repo'].edge_index.shape[1]} edges\n\n")
        
        f.write("Edge Attributes:\n")
        f.write(f"  - log_weight (log-transformed active days)\n")
        f.write(f"  - Shape: {data['user', 'contributes_to', 'repo'].edge_attr.shape}\n\n")
        
        f.write("Edge Weight Statistics:\n")
        weights = data['user', 'contributes_to', 'repo'].edge_attr.squeeze()
        f.write(f"  - Min: {weights.min().item():.4f}\n")
        f.write(f"  - Max: {weights.max().item():.4f}\n")
        f.write(f"  - Mean: {weights.mean().item():.4f}\n")
        f.write(f"  - Std: {weights.std().item():.4f}\n\n")
        
        f.write("Graph Density:\n")
        num_users = len(mappings['user_to_idx'])
        num_repos = len(mappings['repo_to_idx'])
        num_edges = data['user', 'contributes_to', 'repo'].edge_index.shape[1]
        max_possible_edges = num_users * num_repos
        density = num_edges / max_possible_edges
        f.write(f"  - Actual edges: {num_edges:,}\n")
        f.write(f"  - Maximum possible edges: {max_possible_edges:,}\n")
        f.write(f"  - Density: {density:.6f} ({density*100:.4f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("Data Period: January - March 2022 (aggregated)\n")
        f.write("="*80 + "\n")
    
    print(f"  Saved to: {output_path}")


def main():
    print("="*80)
    print("CREATING PYTORCH GEOMETRIC HETERODATA")
    print("="*80)
    print("Data period: January - March 2022 (aggregated)")
    print()
    
    # Load features
    user_df, repo_df = load_features(USER_FEATURES_PARQUET, REPO_FEATURES_PARQUET)
    
    # Create node mappings
    mappings = create_node_mappings(user_df, repo_df)
    
    # Load edges
    print("\nLoading edges...")
    edges_df = pd.read_parquet(TRANSFORMED_EDGES_PARQUET)
    print(f"  Loaded {len(edges_df)} edges")
    print(f"  Using 'log_weight' for edge attributes")
    
    # Build HeteroData
    data = build_hetero_data(user_df, repo_df, edges_df, mappings)
    
    # Save HeteroData
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    print(f"\nSaving HeteroData to {HETERO_DATA_OUTPUT}...")
    torch.save(data, HETERO_DATA_OUTPUT)
    print("  ✓ HeteroData saved")
    
    # Save node mappings
    print(f"\nSaving node mappings to {NODE_MAPPINGS_OUTPUT}...")
    with open(NODE_MAPPINGS_OUTPUT, 'wb') as f:
        pickle.dump(mappings, f)
    print("  ✓ Node mappings saved")
    
    # Save graph info
    save_graph_info(data, mappings, GRAPH_INFO_OUTPUT)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nHeteroData structure:")
    print(data)
    
    print("\nOutputs created:")
    print(f"  1. HeteroData: {HETERO_DATA_OUTPUT}")
    print(f"  2. Node mappings: {NODE_MAPPINGS_OUTPUT}")
    print(f"  3. Graph info: {GRAPH_INFO_OUTPUT}")
    
    print("\nNode mappings contain:")
    print("  - user_to_idx: dict mapping username -> node index")
    print("  - idx_to_user: dict mapping node index -> username")
    print("  - repo_to_idx: dict mapping repo name -> node index")
    print("  - idx_to_repo: dict mapping node index -> repo name")
    
    print("\nNext steps:")
    print("  1. Load HeteroData for GNN training:")
    print("     data = torch.load('hetero_graph.pt')")
    print("  2. Load mappings for evaluation:")
    print("     with open('node_mappings.pkl', 'rb') as f:")
    print("         mappings = pickle.load(f)")
    print("  3. Build and train your GNN model!")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()