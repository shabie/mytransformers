from mytransformers.downloaders import fetch_model
files_to_save = None
url = "https://huggingface.co/ankur310794/roberta-base-squad2-nq"

fetch_model(url, download_dir=".", files_to_save=files_to_save, upload_to_kaggle=True)
