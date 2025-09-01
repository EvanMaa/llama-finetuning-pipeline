import os
from data_collection_scripts import scrape_and_prepare_single_pass
from model_training_scripts import train_on_sub, chat_with_sub

def main(sub):
    data_path_train = f"Project-1/data/raw/{sub}_train.jsonl"
    data_path_val = f"Project-1/data/raw/{sub}_val.jsonl"
    data_path_test = f"Project-1/data/raw/{sub}_test.jsonl"

    if os.path.exists(data_path_train) and os.path.exists(data_path_val) and os.path.exists(data_path_test):
        print(f"[INFO] Dataset for r/{sub} already exists. Skipping scraping.")
    else:
        scrape_and_prepare_single_pass(sub, 10000)

    model_path =  f"Project-1/models/{sub}"

    if os.path.exists(model_path):
        print(f"[INFO] Model for r/{sub} already exists. Skipping training.")
    else:
        train_on_sub(sub)

    chat_with_sub(sub)

if __name__ == "__main__":
    sub = input("Enter subreddit to use: ").strip()
    main(sub)
    main(sub)