# scraper.py
import os
from dotenv import load_dotenv
import praw
import prawcore
import time
import sys

def init_reddit():
    load_dotenv()
    reddit = praw.Reddit(
        
        # Essentially just API keys
        # we don't do load_dotenv(dotenv_path="") because only 1 env file
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),

        # reddit identifier - reddit tracks who wants to use API (reddify)
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
    reddit.read_only = True

    return(reddit)


def safe_fetch(generator, mode_name):
    """
    Wraps a subreddit listing generator so it retries indefinitely
    if we hit rate limiting (HTTP 429) or temporary errors.
    """
    while True:
        try:
            for post in generator:
                yield post
            break  # exit once generator completes
        except prawcore.exceptions.TooManyRequests as e:
            wait_time = getattr(e, "sleep_time", 60)
            print(f"[WARNING] Rate limited while fetching {mode_name}. Sleeping {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] Unexpected error in {mode_name}: {e}. Sleeping 30s before retry...")
            time.sleep(30)


def fetch_from_mode(subreddit, mode, target_limit, every_post, seen_ids):
    posts_fetched = 0
    while posts_fetched < target_limit:
        try:
            fetch_fn = getattr(subreddit, mode)  # e.g., subreddit.top
            for post in fetch_fn(limit=None):    # now definitely iterable
                if post.id in seen_ids:
                    continue
                post.comments.replace_more(limit=0)
                first_comment = post.comments[0] if post.comments else None
                if not first_comment:
                    continue
                every_post.append({
                    "title": post.title,
                    "body": post.selftext,
                    "reply": first_comment.body
                })
                seen_ids.add(post.id)
                posts_fetched += 1
                if len(every_post) >= target_limit:
                    return every_post
            break  # exit loop if no error
        except prawcore.exceptions.TooManyRequests as e:
            wait_time = getattr(e, "sleep_time", 30)  # use Reddit’s suggested wait or default
            print(f"[RATE LIMIT] Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] Unexpected error in {mode}: {e}. Stopping...")
            return
    return every_post


# fetch_subreddit_posts() --> (PRAW reddit instance, subreddit needed, limit *minimum set to 10)
def fetch_subreddit_posts(reddit, subreddit_name, limit=10):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Test if subreddit exists by trying to fetch one post
        _ = next(subreddit.top(limit=1))
    except:
        print(f"Subreddit not found or inaccessible: {subreddit_name}")
        sys.exit(1)
        return []
        

    every_post = []
    seen_ids = set()

# Step 1: top posts
    fetch_from_mode(subreddit, "top", limit, every_post, seen_ids)
    # Step 2: controversial if not enough
    if len(every_post) < limit:
        fetch_from_mode(subreddit, "controversial", limit, every_post, seen_ids)
    # Step 3: hot if still not enough
    if len(every_post) < limit:
        fetch_from_mode(subreddit, "hot", limit, every_post, seen_ids)

    return every_post


def main():
    reddit = init_reddit()
    user_subreddit = input("Enter a subreddit to scrape: ")
    limit = int(input("How many posts to fetch? "))
    every_post = fetch_subreddit_posts(reddit, user_subreddit, limit)

    print(f"\n\nFetched {len(every_post)} posts from r/{user_subreddit}")

    for p in every_post:
        print(p)
        print("-" * 50)

# __name__ is a built in variable, checks if exact file is being run or it's being called
if __name__ == "__main__":
    main()