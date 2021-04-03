from mytransformers.downloaders import fetch_model


# set to None for all files
files_to_save = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
]

url = "https://huggingface.co/ankur310794/roberta-base-squad2-nq"

fetch_model(url, download_dir=".", files_to_save=files_to_save, upload_to_kaggle=True)
