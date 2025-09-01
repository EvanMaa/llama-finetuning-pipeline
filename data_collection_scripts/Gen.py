from .full_pipeline import scrape_and_prepare_single_pass
from .scraper import init_reddit
import time

# List of subreddits you want to scrape
subreddits = [ 
    "summonerschool",
    "OntarioGrade12s",
    "teenagers",
    "AskWomen",
    "LocalLLaMA",
]


for sub in subreddits:
    try:
        scrape_and_prepare_single_pass(sub, total_data_points=5000)
        # Optional delay to avoid hitting Reddit rate limits
        time.sleep(60)
    except Exception as e:
        print(f"[ERROR] Failed to scrape r/{sub}: {e}")