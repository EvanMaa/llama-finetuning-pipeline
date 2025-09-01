import time
from .scraper import init_reddit, fetch_subreddit_posts
from .prepare_dataset_jsonl import clean_posts, prepare_jsonl_dataset

def scrape_and_prepare_single_pass(subreddit_name, total_data_points=2000):
    reddit = init_reddit()

    start_time = time.time()
    print(f"[DEBUG] Fetching posts from r/{subreddit_name}...")

    # Fetch extra posts to account for skipped posts
    raw_posts = fetch_subreddit_posts(
        reddit,
        subreddit_name,
        limit=total_data_points * 2  # fetch extra to ensure enough valid posts
    )

    print(f"[DEBUG] Fetched {len(raw_posts)} raw posts in {time.time() - start_time:.2f}s")

    # Clean posts and remove posts with empty bodies or comments
    cleaned = clean_posts(raw_posts)

    if not cleaned:
        print("[WARNING] No valid posts found. Exiting.")
        return

    # Limit to total_data_points if more were fetched
    if len(cleaned) > total_data_points:
        cleaned = cleaned[:total_data_points]

    print(f"[DEBUG] {len(cleaned)} posts after cleaning")

    # Save dataset into train/val/test JSONL
    prepare_jsonl_dataset(cleaned, base_filename=subreddit_name, subreddit_name=subreddit_name)
    print(f"[DEBUG] Finished processing r/{subreddit_name} in {time.time() - start_time:.2f}s")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    subreddit_input = input("Enter subreddit to scrape: ").strip()
    scrape_and_prepare_single_pass(subreddit_input, total_data_points=10)
