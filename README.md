LLaMA Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning large language models (LLMs) on Reddit data, including data collection, preprocessing, training, and interfacing.

Overview

This repository provides a complete workflow for fine-tuning LLMs on domain-specific Reddit data. The pipeline covers:

* Data scraping and cleaning
* Dataset preparation for training, validation, and testing
* Parameter-efficient fine-tuning (LoRA / PEFT)
* It demonstrates practical skills in NLP, deep learning, and MLOps, and can be used as a foundation for custom LLM adaptation projects.

Features

* Automated Reddit data collection and filtering
* JSONL dataset generation for model training
* Tokenization and preprocessing pipeline
* Training scripts

Tech Stack

This project brings together tools from deep learning, data engineering, and API integration:

Python 3.11+
 – Core programming language
PyTorch
 – Deep learning framework for model fine-tuning
Hugging Face Transformers
 – Pretrained LLaMA and utilities for tokenization, training, and inference
PEFT / LoRA
 – Parameter-efficient fine-tuning techniques for large models
 BitsAndBytes
 – 8-bit/4-bit model quantization for memory-efficient training on limited GPU resources
PRAW
 – Python Reddit API Wrapper for data scraping
python-dotenv
 – Environment variable management for secure API credentials
Datasets
 – Efficient dataset loading and preprocessing
SQLite / JSONL
 – Lightweight storage of collected Reddit data
 
Environment Setup and Reddit API
* This project requires access to the Reddit API to scrape data from subreddits. You will need to create a .env file to securely store your credentials.
* Look at requirements.txt for all required libraries, note: this is configured for an rtx 5080 depending on your gpu or cpu this may run very slowly.

Example Prompts:

---

(trained on askprogramming)

You: Explain to me the difference between a compiler and interpreter

Assistant: A compiler is a program that translates code from one programming language into another. It takes the source code, which is written in one language, and produces an executable file in another language that can be run on a computer. The most common example of this is when a compiler takes compiled languages like C or Java and compiles them into machine code that can run on any computer.
An interpreter, on the other hand, is a program that executes or interprets code line by line instead of compiling it ahead of time. It takes the source code and performs the commands directly without converting it into another format. This means that the code needs to be readable by the interpreter at all times; it cannot be encrypted or obfuscated.
One of the main differences between compilers and interpreters is how they handle errors. When a compiler encounteres an error in the code, it will usually generate an error message and stop processing the rest of the code until the error is fixed. An interpreter, on the other hand, will continueto execute the code up until the point of the error, and then stop and display an error message.

---

(train on amitheasshole)

You: AITA for asking my friend to pay me back for food I cooked for her dinner party? I (29F) love cooking and often host dinners for friends. Recently, my friend (31F) asked me to cook for a party she was hosting at her house because she loves my food and wanted to impress her guests I agreed but told her I would need her to cover the cost of ingredients since it wasn’t my event and She seemed fine with it. 
After the party, I gave her the receipt around $150 and thats just ingredients I didn't take any money for myself , She said she didn’t expect it to cost that much and that I should cover it as a gift because we’re friends and I enjoy cooking I said no and now she’s calling me tacky for "nickel and diming” her over dinner. AITA for asking her to pay me back?

Assistant: nobody is forcing you to do this. if you don't want to make or share your food, then DON"T! simple as that.
i do think it's kinda weird though that yor friend wants you to make and pay for your own food when hosting an entire event. that doesn't seem very fair...but maybe they are trying to be nice by making up for it by throwing in some extra cash? i wouldn't know.

---

(trained on datascience)

You: What is a tensor?

Assistant: A tensor is a mathematical object that can be used to represent data in a multi-dimensional array. It's similar to a matrix, but a more flexible and generalization thereof. Tensors are used widely in machine learning and other fields where data has multiple dimensions. For example, you could use a 3D tensor to represent an image, with each "pixel" having three values for red, green, and blue. 




