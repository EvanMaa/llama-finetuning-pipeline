# prepare_dataset_jsonl.py
import os
import json
import random
from .process_dataset import clean_posts

def split_dataset(posts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into training, validation, and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    total = len(posts)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_set = posts[:train_end]
    val_set = posts[train_end:val_end]
    test_set = posts[val_end:]
    
    return train_set, val_set, test_set

def save_jsonl(posts, filename, subreddit_name="llama_dataset"):
    """
    Save posts in LLaMA-ready JSONL format with system/user/assistant.
    Each post must have 'title', 'body', and 'reply'.
    """
    os.makedirs("./data/raw", exist_ok=True)
    filepath = os.path.join("./data/raw", filename)

    system_prompt = f"You are a typical Reddit user from r/{subreddit_name}. Respond casually in subreddit style."
    
    with open(filepath, "w", encoding="utf-8") as f:
        for post in posts:
            user_text = f"{post['title']}\n\n{post['body']}"
            json_line = {
                "system": system_prompt,
                "user": user_text,
                "assistant": post["reply"]
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(posts)} posts to {filepath}")
    return filepath

def prepare_jsonl_dataset(raw_posts, base_filename="llama_dataset", subreddit_name="llama_dataset"):
    """
    Clean raw posts, shuffle, split into train/val/test, and save JSONL files.
    """
    cleaned = clean_posts(raw_posts)
    random.shuffle(cleaned)
    
    train_set, val_set, test_set = split_dataset(cleaned)
    
    save_jsonl(train_set, filename=f"{base_filename}_train.jsonl", subreddit_name=subreddit_name)
    save_jsonl(val_set, filename=f"{base_filename}_val.jsonl", subreddit_name=subreddit_name)
    save_jsonl(test_set, filename=f"{base_filename}_test.jsonl", subreddit_name=subreddit_name)
    
    print(f"Dataset split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")


if __name__ == "__main__":
    raw_posts = [
        {"title": "Post 1", "body": "This is the body of post 1", "reply": "Comment 1"},
        {"title": "Post 2", "body": "Another text post", "reply": "Comment 2"},
        {"title": "Post 3", "body": "More content here", "reply": "Comment 3"}
    ]
    
    prepare_jsonl_dataset(raw_posts, base_filename="example_dataset", subreddit_name="example_subreddit")
