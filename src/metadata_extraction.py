"""
Fetch GitHub metadata for users and repos from filtered parquet file
Uses GitHub GraphQL API with multiple tokens for rate limit handling
"""

import pandas as pd
import json
import requests
from time import sleep
from datetime import datetime

GITHUB_TOKENS = [

]

PARQUET_INPUT = '/Users/hnasrolahi/Desktop/cs782project/filtered_edges_10plus.parquet'
USER_METADATA_OUTPUT = '/Users/hnasrolahi/Desktop/cs782project/user_metadata.json'
REPO_METADATA_OUTPUT = '/Users/hnasrolahi/Desktop/cs782project/repo_metadata.json'

SAVE_FREQUENCY = 50

RATE_LIMIT_BUFFER = 4500

class GitHubAPIClient:
    """Handle GitHub GraphQL API requests with multiple tokens"""
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_idx = 0
        self.requests_made = 0
        self.graphql_url = "https://api.github.com/graphql"
    
    def get_current_token(self):
        return self.tokens[self.current_token_idx]
    
    def switch_token(self):
        """Switch to next token when rate limit is approaching"""
        self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
        self.requests_made = 0
        print(f"\n  → Switching to token {self.current_token_idx + 1}/{len(self.tokens)}")
        sleep(2)  # Brief pause when switching tokens
    
    def execute_query(self, query, variables=None):
        """Execute GraphQL query with automatic token switching"""
        
        # Switch token if approaching rate limit
        if self.requests_made >= RATE_LIMIT_BUFFER:
            print(f"  → Rate limit approaching for token {self.current_token_idx + 1}, switching...")
            self.switch_token()
        
        headers = {
            "Authorization": f"Bearer {self.get_current_token()}",
            "Content-Type": "application/json"
        }
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = requests.post(self.graphql_url, json=payload, headers=headers, timeout=30)
            self.requests_made += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for errors in response
                if "errors" in data:
                    return None, f"GraphQL Error: {data['errors']}"
                
                return data.get("data"), None
            
            elif response.status_code == 401:
                return None, "Invalid token"
            
            elif response.status_code == 403:
                # Rate limited - switch token
                print(f"  → Rate limited on token {self.current_token_idx + 1}")
                self.switch_token()
                return self.execute_query(query, variables)  # Retry with new token
            
            else:
                return None, f"HTTP {response.status_code}: {response.text}"
        
        except Exception as e:
            return None, f"Request failed: {str(e)}"


def fetch_user_metadata(client, username):
    """Fetch metadata for a single user"""
    
    query = """
    query($login: String!) {
        user(login: $login) {
            login
            name
            bio
            company
            location
            createdAt
            followers {
                totalCount
            }
            repositories(privacy: PUBLIC) {
                totalCount
            }
        }
    }
    """
    
    variables = {"login": username}
    data, error = client.execute_query(query, variables)
    
    if error:
        return None
    
    if data and data.get("user"):
        user = data["user"]
        return {
            "login": user.get("login"),
            "name": user.get("name"),
            "bio": user.get("bio"),
            "company": user.get("company"),
            "location": user.get("location"),
            "createdAt": user.get("createdAt"),
            "totalFollowers": user.get("followers", {}).get("totalCount", 0),
            "totalPublicRepos": user.get("repositories", {}).get("totalCount", 0)
        }
    
    return None


def fetch_repo_metadata(client, owner, repo_name):
    """Fetch metadata for a single repository"""
    
    query = """
    query($owner: String!, $name: String!) {
        repository(owner: $owner, name: $name) {
            nameWithOwner
            description
            isArchived
            createdAt
            pushedAt
            stargazerCount
            forkCount
            watchers {
                totalCount
            }
            primaryLanguage {
                name
            }
            repositoryTopics(first: 20) {
                nodes {
                    topic {
                        name
                    }
                }
            }
        }
    }
    """
    
    variables = {"owner": owner, "name": repo_name}
    data, error = client.execute_query(query, variables)
    
    if error:
        return None
    
    if data and data.get("repository"):
        repo = data["repository"]
        
        # Extract topics
        topics = []
        if repo.get("repositoryTopics"):
            for node in repo["repositoryTopics"].get("nodes", []):
                if node.get("topic"):
                    topics.append(node["topic"]["name"])
        
        return {
            "repo_name": repo.get("nameWithOwner"),
            "description": repo.get("description"),
            "isArchived": repo.get("isArchived", False),
            "createdAt": repo.get("createdAt"),
            "pushedAt": repo.get("pushedAt"),
            "stargazerCount": repo.get("stargazerCount", 0),
            "forkCount": repo.get("forkCount", 0),
            "watchCount": repo.get("watchers", {}).get("totalCount", 0),
            "primaryLanguage": repo.get("primaryLanguage", {}).get("name") if repo.get("primaryLanguage") else None,
            "topics": topics
        }
    
    return None


def load_existing_data(filepath):
    """Load existing metadata if file exists"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {filepath}, starting fresh")
        return []


def save_metadata(data, filepath):
    """Save metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("="*80)
    print("GITHUB METADATA FETCHER")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize API client
    print(f"\nInitializing GitHub API client with {len(GITHUB_TOKENS)} tokens...")
    client = GitHubAPIClient(GITHUB_TOKENS)
    
    # Load parquet file
    print(f"\nLoading edge list from: {PARQUET_INPUT}")
    df = pd.read_parquet(PARQUET_INPUT)
    print(f"Loaded {len(df)} edges")
    
    # Extract unique users and repos
    unique_users = df['user'].unique().tolist()
    unique_repos = df['repo'].unique().tolist()
    
    print(f"\nFound:")
    print(f"  - {len(unique_users)} unique users")
    print(f"  - {len(unique_repos)} unique repos")
    
    # Load existing data (in case we're resuming)
    print("\nChecking for existing metadata...")
    user_metadata = load_existing_data(USER_METADATA_OUTPUT)
    repo_metadata = load_existing_data(REPO_METADATA_OUTPUT)
    
    existing_users = {u['login'] for u in user_metadata if u.get('login')}
    existing_repos = {r['repo_name'] for r in repo_metadata if r.get('repo_name')}
    
    print(f"  Already have: {len(existing_users)} users, {len(existing_repos)} repos")
    
    # Filter out already fetched items
    users_to_fetch = [u for u in unique_users if u not in existing_users]
    repos_to_fetch = [r for r in unique_repos if r not in existing_repos]
    
    print(f"\nNeed to fetch:")
    print(f"  - {len(users_to_fetch)} users")
    print(f"  - {len(repos_to_fetch)} repos")
    
    # Fetch user metadata
    if users_to_fetch:
        print("\n" + "="*80)
        print("FETCHING USER METADATA")
        print("="*80)
        
        for idx, username in enumerate(users_to_fetch, 1):
            print(f"\n[{idx}/{len(users_to_fetch)}] Fetching user: {username}")
            
            metadata = fetch_user_metadata(client, username)
            
            if metadata:
                user_metadata.append(metadata)
                print(f"  ✓ Success: {metadata.get('name', 'N/A')} | {metadata.get('totalFollowers', 0)} followers")
            else:
                print(f"  ✗ Failed: User not found or inaccessible")
            
            # Periodic save
            if idx % SAVE_FREQUENCY == 0:
                print(f"\n  → Saving progress ({len(user_metadata)} users)...")
                save_metadata(user_metadata, USER_METADATA_OUTPUT)
            
            # Rate limiting: small delay between requests
            sleep(0.1)
        
        # Final save for users
        print(f"\n  → Final save ({len(user_metadata)} users)...")
        save_metadata(user_metadata, USER_METADATA_OUTPUT)
    
    # Fetch repo metadata
    if repos_to_fetch:
        print("\n" + "="*80)
        print("FETCHING REPO METADATA")
        print("="*80)
        
        for idx, repo_full_name in enumerate(repos_to_fetch, 1):
            print(f"\n[{idx}/{len(repos_to_fetch)}] Fetching repo: {repo_full_name}")
            
            # Parse owner/repo
            if '/' in repo_full_name:
                owner, repo_name = repo_full_name.split('/', 1)
            else:
                print(f"  ✗ Invalid repo format: {repo_full_name}")
                continue
            
            metadata = fetch_repo_metadata(client, owner, repo_name)
            
            if metadata:
                repo_metadata.append(metadata)
                print(f"  ✓ Success: ⭐ {metadata.get('stargazerCount', 0)} | 🍴 {metadata.get('forkCount', 0)} | Lang: {metadata.get('primaryLanguage', 'N/A')}")
            else:
                print(f"  ✗ Failed: Repo not found or inaccessible")
            
            # Periodic save
            if idx % SAVE_FREQUENCY == 0:
                print(f"\n  → Saving progress ({len(repo_metadata)} repos)...")
                save_metadata(repo_metadata, REPO_METADATA_OUTPUT)
            
            # Rate limiting: small delay between requests
            sleep(0.1)
        
        # Final save for repos
        print(f"\n  → Final save ({len(repo_metadata)} repos)...")
        save_metadata(repo_metadata, REPO_METADATA_OUTPUT)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal counts:")
    print(f"  Users: {len(user_metadata)}/{len(unique_users)} ({len(user_metadata)/len(unique_users)*100:.1f}%)")
    print(f"  Repos: {len(repo_metadata)}/{len(unique_repos)} ({len(repo_metadata)/len(unique_repos)*100:.1f}%)")
    
    print(f"\nOutputs saved to:")
    print(f"  - User metadata: {USER_METADATA_OUTPUT}")
    print(f"  - Repo metadata: {REPO_METADATA_OUTPUT}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()