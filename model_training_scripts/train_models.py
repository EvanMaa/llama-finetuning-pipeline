from train import train_on_sub
import time

# List of subreddits you want to scrape
subreddits = [
    "amitheasshole"
]


for sub in subreddits:
    try:
        train_on_sub(sub)
        # Optional delay to avoid hitting Reddit rate limits
        time.sleep(60)
    except Exception as e:
        print(f"[ERROR] Failed to train on r/{sub}: {e}")