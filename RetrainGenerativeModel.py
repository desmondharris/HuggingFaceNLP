from praw import Reddit
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, TFAutoModelForCausalLM, create_optimizer, AdamWeightDecay
import huggingface_hub
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
import re


# Function used is from ActiveState
# https://code.activestate.com/recipes/81330-single-pass-multiple-replace/
def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)



def reddit_to_text(api: Reddit, subreddit_name: str, filepath: str, num_posts: int):
    """Take top posts from a subreddit, and write their body text to a new file.

    Args:
        api (Reddit): Instance of Python Reddit API Wrapper
        subreddit_name (str): Display name of the subreddit
        filepath (str): Where the new file should be located relative to current working directory
        num_posts (int): How many posts to scrape.
    """ 

    subreddit = api.subreddit(subreddit_name)
    print(f"Fetching top {num_posts} posts")
    top_posts = subreddit.top(limit=num_posts)
    print("All posts fetched.")

    mistake_count = 0
    replace_dict = {
        '[': "",
        ']': "",
        "(": "",
        ")": "",
        # ".": "",
        "mr.": "mr",
        "mrs.": "mrs",
        "ms.": "ms",
        "#": "",
        ":": "",
        "@": "",
        "they're": "they are",
        "what's": "what is",
        "let's": "let us",
        "there've": "there have",
        "i've": "i have",
        "don't": "do not",
        "we'd": "we had",
        "he'd": "he had",
        "we're": "we are",
        "they're": "they are",
        "is'nt": "is not",
        "that's": "that is",
        "it's": "it is",
        "he's": "he is",
        "wasn't": "was not",
        "i'd": "i would",
        "you've": "you have",
        "you're": "you are",
        "you'd": "you would",
        "i'm": "i am",
        "we've": "we have",
        '"': "",
        "&": "",
        '\n': " ",
        '<': "",
        '>': "",
        "~": "",
        ".*": "",
        "*": ""
    }
    with open(filepath, 'w') as file:
        for i, post in enumerate(top_posts):
            print(f"\rProcessing post {i+1} of {num_posts}", end="")
            text = post.selftext.lower()
            # RegEx from Lee Martin on Stack Overflow https://stackoverflow.com/users/3960192/lee-martin
            text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
            if text != "":
                file.write(multiple_replace(replace_dict, text))
                file.write('\n')
            else:
                mistake_count += 1
        file.close()


def text_to_data_dict(file: str):
    return load_dataset("text", data_files=file)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
if __name__ == "__main__":


    api = Reddit(client_id="uMo68bhbI96WL8eE2I7h_g",
                        client_secret="xqgkfC3ss8_8a6SHC7eTZwUKS3C_XA",
                        user_agent="SecondBot")

    max_posts = 116
    reddit_to_text(api, "PoliticalDiscussion", "first_test.txt", max_posts)

    cutoff = round((max_posts / 10) * 8)

    train_file = open("train_split.txt", 'w')
    valid_file = open("valid_split.txt", 'w')
    main_file = open("first_test.txt", 'r')

    for i, line in enumerate(main_file):
        if i < cutoff:
            train_file.write(line)
        else:
            valid_file.write(line)
    
    train_file.close()
    valid_file.close()
    main_file.close()

    data_files={
        'train': "train_split.txt",
        'validation': "valid_split.txt"
    }
    dataset = load_dataset("text", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    def preprocces_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_data = dataset.map(
        preprocces_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    block_size = 128

    grouped_data = tokenized_data.map(
        group_texts,
        batched=True,
        num_proc=1
    )

    tf_train_set = model.prepare_tf_dataset(
        grouped_data["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        grouped_data["validation"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)
    huggingface_hub.login(token="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")
    callback = PushToHubCallback(
        output_dir="teshuggingface_hub.logit-clm",
        tokenizer=tokenizer,
        hub_token="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"
    )   

    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])