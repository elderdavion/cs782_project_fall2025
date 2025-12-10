import gzip
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import re

START_DATE = '2022-01-01'
END_DATE = '2022-03-31'

CSV_PATH = '/Users/hnasrolahi/Desktop/cs782project/pypi_github_packages.csv'
GHARCHIVE_BASE = '/Volumes/hamedadata/GHArchive'
OUTPUT_PATH = '/Users/hnasrolahi/Desktop/cs782project/push_events_output.parquet'

SAVE_FREQUENCY = 7

# Bot-related patterns to filter out
BOT_KEYWORDS = [
    'bot', 'dependabot', 'renovate', 'github-actions', 
    'snyk-bot', 'codecov', 'semantic-release', 'pre-commit-ci',
    'greenkeeper', 'imgbot', 'stalebot', 'mergify',
    'allcontributors', 'dependabot-preview', 'pyup-bot',
    'travis', 'circleci', 'jenkins'
]

BOT_EMAIL_DOMAINS = [
    '@users.noreply.github.com',
    '@github.com',
    '@greenkeeper.io',
    '@renovatebot.com',
    '@snyk.io'
]

def is_bot(login, author_email=None):
    # Check login for bot keywords (case-insensitive)
    login_lower = login.lower()
    for keyword in BOT_KEYWORDS:
        if keyword in login_lower:
            return True
    
    # Check email domain if provided
    if author_email:
        author_email_lower = author_email.lower()
        for domain in BOT_EMAIL_DOMAINS:
            if domain in author_email_lower:
                return True
    
    # Check for [bot] suffix (GitHub's official bot marker)
    if login.endswith('[bot]'):
        return True
    
    return False


def load_repo_list(csv_path):
    print(f"Loading repository list from {csv_path}...")
    df = pd.read_csv(csv_path)
    repos = set()
    for repo in df['github_repo'].dropna():
        if isinstance(repo, str):
            if 'github.com/' in repo:
                repo = repo.split('github.com/')[-1]
            repo = repo.rstrip('.git').rstrip('/')
            repos.add(repo.lower())
    
    print(f"Loaded {len(repos)} repositories")
    return repos


def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def get_file_path(base_path, date, hour):
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    
    filename = f"{year}-{month}-{day}-{hour}.json.gz"
    return Path(base_path) / year / month / day / filename


def process_push_event(event, repo_set):
    # Check if it's a PushEvent
    if event.get('type') != 'PushEvent':
        return []
    
    # Get repo name and normalize
    repo_name = event.get('repo', {}).get('name', '').lower()
    if not repo_name or repo_name not in repo_set:
        return []
    
    # Check distinct_size
    payload = event.get('payload', {})
    distinct_size = payload.get('distinct_size', 0)
    if distinct_size < 1:
        return []
    
    # Get actor login
    actor = event.get('actor', {})
    login = actor.get('login', '')
    if not login:
        return []
    
    # Get commit author email(s) for additional bot detection
    commits = payload.get('commits', [])
    author_emails = [c.get('author', {}).get('email', '') for c in commits]
    primary_email = author_emails[0] if author_emails else None
    
    # Filter out bots
    if is_bot(login, primary_email):
        return []
    
    # Get the date (without time) for tracking active days
    created_at = event.get('created_at', '')
    try:
        event_date = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ').date()
    except:
        return []
    
    return [(login, repo_name, event_date, primary_email)]

def process_file(file_path, repo_set):
    results = []
    
    if not file_path.exists():
        print(f"  Warning: File not found: {file_path}")
        return results
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    results.extend(process_push_event(event, repo_set))
                except json.JSONDecodeError:
                    continue  # Skip malformed JSON lines
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
    
    return results


def aggregate_results(raw_data):
    aggregated = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'dates': set()}))
    
    for login, repo, date, _ in raw_data:
        aggregated[login][repo]['count'] += 1  # Count each PushEvent
        aggregated[login][repo]['dates'].add(date)  # Track unique dates
    
    # Convert to list of records
    records = []
    for login, repos in aggregated.items():
        for repo, data in repos.items():
            records.append({
                'login': login,
                'repo': repo,
                'frequency': data['count'],  # Total number of PushEvents
                'active_days': len(data['dates'])  # Number of unique days
            })
    
    return pd.DataFrame(records)


def save_results(df, output_path, mode='append'):
    if df.empty:
        print("  No data to save")
        return
    
    output_file = Path(output_path)
    
    if mode == 'append' and output_file.exists():
        # Read existing data and combine
        existing_df = pd.read_parquet(output_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        
        # Re-aggregate to handle duplicates
        # Sum frequencies (total PushEvents) and active_days
        aggregated = combined_df.groupby(['login', 'repo']).agg({
            'frequency': 'sum',  # Sum all PushEvents
            'active_days': 'sum'  # Sum all unique days
        }).reset_index()
        
        aggregated.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"  Appended {len(df)} records (total: {len(aggregated)} unique combinations)")
    else:
        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"  Saved {len(df)} records to {output_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("GitHub Archive PushEvent Extractor")
    print("=" * 70)
    print(f"Time window: {START_DATE} to {END_DATE}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    
    # Load repository list
    repo_set = load_repo_list(CSV_PATH)
    
    # Storage for accumulated data
    accumulated_data = []
    days_processed = 0
    
    # Process each date in range
    dates = list(generate_date_range(START_DATE, END_DATE))
    total_days = len(dates)
    
    print(f"\nProcessing {total_days} days of data...")
    print()
    
    for date_idx, date in enumerate(dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        print(f"[{date_idx}/{total_days}] Processing {date_str}...")
        
        day_data = []
        
        # Process all 24 hours for this date
        for hour in range(24):
            file_path = get_file_path(GHARCHIVE_BASE, date, hour)
            hour_results = process_file(file_path, repo_set)
            day_data.extend(hour_results)
        
        print(f"  Found {len(day_data)} matching PushEvents")
        accumulated_data.extend(day_data)
        days_processed += 1
        
        # Periodic save to avoid memory issues
        if days_processed % SAVE_FREQUENCY == 0 or date_idx == total_days:
            print(f"\n  Saving accumulated data ({len(accumulated_data)} raw events)...")
            df = aggregate_results(accumulated_data)
            save_results(df, OUTPUT_PATH, mode='append' if date_idx > SAVE_FREQUENCY else 'overwrite')
            accumulated_data = []  # Clear memory
            print()
    
    print("=" * 70)
    print("Processing complete!")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()