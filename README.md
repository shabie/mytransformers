# My Transformers

This is a very small package to download HuggingFace transformer model repos to disk. It can also
be used to upload the models to Kaggle (which is handy for kernel competitions).

Apart from uploading to Kaggle, this has the advantage that models are not loaded to memory when being downloaded.

## Installation

`pip install git+ssh://git@github.com/shabie/mytransformers.git`

## Usage
It only consists of one function really and relies on APIs of `transformers` libary and `kaggle`
api.

For using Kaggle, it is important to have `kaggle.json` setup already.

```python
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
```

The example above can be found in the folder `examples`.


![example of fetch model](examples/example.svg)

## License
MIT
