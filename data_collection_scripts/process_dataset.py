# process_dataset.py
import os
import json

def clean_posts(raw_posts):
    """
    Clean a list of raw Reddit posts.
    Each comment is treated as a separate data point with 'reply'.
    Skip posts or comments that are empty.
    """
    cleaned = []
    for post in raw_posts:
        title = post.get("title", "").strip()
        body = post.get("body", "").strip()
        reply = post.get("reply", "").strip()  # comment
        
        # Skip if body or reply is empty
        if not body or not reply:
            continue
        
        cleaned.append({
            "title": title,
            "body": body,
            "reply": reply
        })
    
    return cleaned


def save_to_jsonl(posts, filename=None, subreddit_name="llama_dataset"):
    """
    Save a list of posts to JSONL in LLaMA-ready format.
    Each post must have 'title', 'body', and 'reply'.
    """
    os.makedirs("./data", exist_ok=True)

    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{subreddit_name}_{timestamp}.jsonl"

    filepath = os.path.join("./data", filename)

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


if __name__ == "__main__":
    raw_posts = [
        {"title": "Post 1", "body": "Body 1", "reply": "Comment 1"},
        {"title": "Post 2", "body": "Body 2", "reply": ""},
        {"title": "Post 3", "body": "Body 3", "reply": "Comment 3"}
    ]

    cleaned = clean_posts(raw_posts)
    save_to_jsonl(cleaned, subreddit_name="example_subreddit")
