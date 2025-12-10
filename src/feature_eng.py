"""
Feature Engineering for Users and Repos
Creates node features using sentence embeddings and metadata
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================== CONFIGURATION ===========================
# Input files
USER_METADATA_JSON = '/Users/hnasrolahi/Desktop/cs782project/user_metadata.json'
REPO_METADATA_JSON = '/Users/hnasrolahi/Desktop/cs782project/repo_metadata.json'
EDGE_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/filtered_edges_10plus.parquet'

# Output files
USER_FEATURES_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/user_features.parquet'
REPO_FEATURES_PARQUET = '/Users/hnasrolahi/Desktop/cs782project/repo_features.parquet'

# Sentence transformer model
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Reference date for calculating experience (end of 2021)
REFERENCE_DATE = datetime(2021, 12, 31)
# =====================================================================

def load_json(filepath):
    """Load JSON file"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} entries")
    return data


def calculate_months_experience(created_at_str, reference_date):
    """
    Calculate months of experience from creation date to reference date.
    Returns 0 if created_at is invalid or after reference date.
    """
    if not created_at_str:
        return 0
    
    try:
        created_at = datetime.strptime(created_at_str, '%Y-%m-%dT%H:%M:%SZ')
        
        if created_at > reference_date:
            return 0
        
        months = (reference_date.year - created_at.year) * 12 + (reference_date.month - created_at.month)
        return max(0, months)
    except:
        return 0


def create_user_features(user_metadata, users_in_graph, model):
    """
    Create feature vectors for users.
    
    Features:
    - Bio embedding (384 dims)
    - Log(experience in months + 1)
    - Log(total followers + 1)
    - Log(total public repos + 1)
    
    Total: 387 dimensions
    """
    print("\n" + "="*80)
    print("CREATING USER FEATURES")
    print("="*80)
    
    # Create lookup dictionary
    user_dict = {u['login']: u for u in user_metadata if u.get('login')}
    
    print(f"\nProcessing {len(users_in_graph)} users...")
    
    # Prepare data
    user_data = []
    bios = []
    
    for username in users_in_graph:
        user = user_dict.get(username, {})
        
        # Extract bio
        bio = user.get('bio') or ""
        bios.append(bio)
        
        # Calculate experience
        created_at = user.get('createdAt')
        experience_months = calculate_months_experience(created_at, REFERENCE_DATE)
        
        # Extract numerical features
        followers = user.get('totalFollowers', 0)
        repos = user.get('totalPublicRepos', 0)
        
        user_data.append({
            'user': username,
            'experience_months': experience_months,
            'followers': followers,
            'repos': repos
        })
    
    # Create bio embeddings
    print("\n  Encoding user bios...")
    bio_embeddings = model.encode(
        bios,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"  Created embeddings shape: {bio_embeddings.shape}")
    
    # Create DataFrame
    df = pd.DataFrame(user_data)
    
    # Apply log transformation
    print("\n  Applying log transformation to numerical features...")
    df['log_experience'] = np.log1p(df['experience_months'])
    df['log_followers'] = np.log1p(df['followers'])
    df['log_repos'] = np.log1p(df['repos'])
    
    print(f"\n  Numerical feature statistics:")
    print(f"    Experience (months): min={df['experience_months'].min()}, max={df['experience_months'].max()}, mean={df['experience_months'].mean():.1f}")
    print(f"    Followers: min={df['followers'].min()}, max={df['followers'].max()}, mean={df['followers'].mean():.1f}")
    print(f"    Repos: min={df['repos'].min()}, max={df['repos'].max()}, mean={df['repos'].mean():.1f}")
    
    # Combine features: bio_embedding + log_experience + log_followers + log_repos
    feature_columns = []
    
    # Add bio embeddings
    for i in range(bio_embeddings.shape[1]):
        df[f'bio_emb_{i}'] = bio_embeddings[:, i]
        feature_columns.append(f'bio_emb_{i}')
    
    # Add log-transformed features
    feature_columns.extend(['log_experience', 'log_followers', 'log_repos'])
    
    # Create final feature dataframe
    result_df = df[['user'] + feature_columns].copy()
    
    print(f"\n  Final user features shape: ({len(result_df)}, {len(feature_columns)})")
    print(f"  Feature dimensions: {len(feature_columns)} (384 bio + 3 numerical)")
    
    return result_df


def get_language_encoding(languages):
    """
    One-hot encode primary languages.
    Returns encoded array and feature names.
    """
    # Get unique languages (excluding None)
    unique_langs = sorted(set(lang for lang in languages if lang))
    unique_langs.append('Unidentified')  # For None values
    
    print(f"\n  Found {len(unique_langs)} unique languages: {unique_langs[:20]}{'...' if len(unique_langs) > 20 else ''}")
    
    # Create one-hot encoding
    lang_to_idx = {lang: idx for idx, lang in enumerate(unique_langs)}
    
    encoded = np.zeros((len(languages), len(unique_langs)))
    
    for i, lang in enumerate(languages):
        if lang is None:
            lang = 'Unidentified'
        encoded[i, lang_to_idx[lang]] = 1
    
    feature_names = [f'lang_{lang}' for lang in unique_langs]
    
    return encoded, feature_names


def create_repo_features(repo_metadata, repos_in_graph, model):
    """
    Create feature vectors for repos.
    
    Features:
    - (Description + topics) embedding (384 dims)
    - Log(stars + 1)
    - Log(forks + 1)
    - Log(watchers + 1)
    - One-hot encoded primary language
    
    Total: ~387 + num_languages dimensions
    """
    print("\n" + "="*80)
    print("CREATING REPO FEATURES")
    print("="*80)
    
    # Create lookup dictionary
    repo_dict = {r['repo_name']: r for r in repo_metadata if r.get('repo_name')}
    
    print(f"\nProcessing {len(repos_in_graph)} repos...")
    
    # Prepare data
    repo_data = []
    combined_texts = []
    languages = []
    
    for repo_name in repos_in_graph:
        repo = repo_dict.get(repo_name, {})
        
        # Combine description and topics
        description = repo.get('description') or ""
        topics = repo.get('topics', [])
        topics_text = " ".join(topics)
        combined = f"{description} {topics_text}".strip()
        
        # Handle completely empty case
        if not combined:
            combined = ""
        
        combined_texts.append(combined)
        
        # Extract numerical features
        stars = repo.get('stargazerCount', 0)
        forks = repo.get('forkCount', 0)
        watches = repo.get('watchCount', 0)
        
        # Extract language
        language = repo.get('primaryLanguage')
        languages.append(language)
        
        repo_data.append({
            'repo': repo_name,
            'stars': stars,
            'forks': forks,
            'watches': watches
        })
    
    # Create description+topics embeddings
    print("\n  Encoding repo descriptions + topics...")
    desc_embeddings = model.encode(
        combined_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"  Created embeddings shape: {desc_embeddings.shape}")
    
    # Create DataFrame
    df = pd.DataFrame(repo_data)
    
    # Apply log transformation
    print("\n  Applying log transformation to numerical features...")
    df['log_stars'] = np.log1p(df['stars'])
    df['log_forks'] = np.log1p(df['forks'])
    df['log_watches'] = np.log1p(df['watches'])
    
    print(f"\n  Numerical feature statistics:")
    print(f"    Stars: min={df['stars'].min()}, max={df['stars'].max()}, mean={df['stars'].mean():.1f}")
    print(f"    Forks: min={df['forks'].min()}, max={df['forks'].max()}, mean={df['forks'].mean():.1f}")
    print(f"    Watches: min={df['watches'].min()}, max={df['watches'].max()}, mean={df['watches'].mean():.1f}")
    
    # One-hot encode languages
    print("\n  One-hot encoding primary languages...")
    lang_encoded, lang_feature_names = get_language_encoding(languages)
    
    # Combine features
    feature_columns = []
    
    # Add description embeddings
    for i in range(desc_embeddings.shape[1]):
        df[f'desc_emb_{i}'] = desc_embeddings[:, i]
        feature_columns.append(f'desc_emb_{i}')
    
    # Add log-transformed features
    feature_columns.extend(['log_stars', 'log_forks', 'log_watches'])
    
    # Add language features
    for i, lang_name in enumerate(lang_feature_names):
        df[lang_name] = lang_encoded[:, i]
        feature_columns.append(lang_name)
    
    # Create final feature dataframe
    result_df = df[['repo'] + feature_columns].copy()
    
    print(f"\n  Final repo features shape: ({len(result_df)}, {len(feature_columns)})")
    print(f"  Feature dimensions: {len(feature_columns)} (384 desc + 3 numerical + {len(lang_feature_names)} languages)")
    
    return result_df


def main():
    print("="*80)
    print("FEATURE ENGINEERING FOR GRAPH NEURAL NETWORK")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load metadata
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    user_metadata = load_json(USER_METADATA_JSON)
    repo_metadata = load_json(REPO_METADATA_JSON)
    
    # Load edge list to get unique users and repos
    print(f"\nLoading edge list from {EDGE_PARQUET}...")
    edges_df = pd.read_parquet(EDGE_PARQUET)
    print(f"  Loaded {len(edges_df)} edges")
    
    users_in_graph = edges_df['user'].unique().tolist()
    repos_in_graph = edges_df['repo'].unique().tolist()
    
    print(f"\n  Unique users: {len(users_in_graph)}")
    print(f"  Unique repos: {len(repos_in_graph)}")
    
    # Apply log transformation to edge weights
    print("\n  Applying log transformation to edge weights...")
    print(f"    Original weight (active_days): min={edges_df['weight'].min()}, max={edges_df['weight'].max()}, mean={edges_df['weight'].mean():.2f}")
    edges_df['log_weight'] = np.log1p(edges_df['weight'])
    print(f"    Log-transformed weight: min={edges_df['log_weight'].min():.2f}, max={edges_df['log_weight'].max():.2f}, mean={edges_df['log_weight'].mean():.2f}")
    
    # Save transformed edges back to parquet
    transformed_edges_output = EDGE_PARQUET.replace('.parquet', '_transformed.parquet')
    edges_df.to_parquet(transformed_edges_output, index=False)
    print(f"  Saved transformed edges to: {transformed_edges_output}")
    
    # Load sentence transformer model
    print("\n" + "="*80)
    print("LOADING SENTENCE TRANSFORMER MODEL")
    print("="*80)
    print(f"Model: {MODEL_NAME}")
    print("This may take a moment on first run (downloading model)...")
    
    model = SentenceTransformer(MODEL_NAME)
    print("✓ Model loaded successfully")
    
    # Create user features
    user_features_df = create_user_features(user_metadata, users_in_graph, model)
    
    # Create repo features
    repo_features_df = create_repo_features(repo_metadata, repos_in_graph, model)
    
    # Save features
    print("\n" + "="*80)
    print("SAVING FEATURES")
    print("="*80)
    
    print(f"\nSaving user features to {USER_FEATURES_PARQUET}...")
    user_features_df.to_parquet(USER_FEATURES_PARQUET, index=False)
    print("✓ User features saved")
    
    print(f"\nSaving repo features to {REPO_FEATURES_PARQUET}...")
    repo_features_df.to_parquet(REPO_FEATURES_PARQUET, index=False)
    print("✓ Repo features saved")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nUser features:")
    print(f"  - Count: {len(user_features_df)}")
    print(f"  - Dimensions: {len(user_features_df.columns) - 1}")  # -1 for user column
    print(f"  - Breakdown: 384 (bio) + 3 (numerical) = 387")
    
    print(f"\nRepo features:")
    print(f"  - Count: {len(repo_features_df)}")
    print(f"  - Dimensions: {len(repo_features_df.columns) - 1}")  # -1 for repo column
    print(f"  - Breakdown: 384 (desc+topics) + 3 (numerical) + {len(repo_features_df.columns) - 388} (languages)")
    
    print(f"\nOutputs:")
    print(f"  - User features: {USER_FEATURES_PARQUET}")
    print(f"  - Repo features: {REPO_FEATURES_PARQUET}")
    print(f"  - Transformed edges: {transformed_edges_output}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Load these parquet files when building your GNN")
    print("  2. Use 'log_weight' column for edge weights (not 'weight')")
    print("  3. Map user/repo names to node indices")
    print("  4. Create feature tensors for PyTorch Geometric")


if __name__ == '__main__':
    main()