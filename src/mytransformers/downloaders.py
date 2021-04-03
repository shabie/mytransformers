import os
from functools import partial
import glob
import json
import re
import requests
import tempfile
import urllib
import warnings

# Downloads
import bs4
from transformers.file_utils import http_get

# visual formatting
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter
from pygments import highlight
from pprint import pformat
from colorama import Fore, Back, Style

# for uploading to kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


def pprint_color(obj):
    print(highlight(pformat(obj), PythonLexer(), Terminal256Formatter()))


SUFFIX = "/tree/main"
BASE_URL = "https://huggingface.co"


def get_correct_url(base_url):
    if not base_url.endswith(SUFFIX):
        base_url = base_url + SUFFIX
    response = requests.head(base_url)
    response.raise_for_status()
    return base_url


def extract_model_id(url):
    assert BASE_URL in url, f"Base URL {BASE_URL} not in URL string: {url}"
    return re.search(r"https://huggingface.co/(.*)" + SUFFIX, url).group(1)


def check_url_status(url):
    response = requests.head(url, allow_redirects=False)
    response.raise_for_status()


def get_tag(element):
    if isinstance(element, bs4.element.NavigableString):
        return -1  # return a number interpreted as missing tag
    tags = element.find_all("a")
    return tags[1]  # 2nd tag contains the download links


def get_file_list(soup):
    page_lists = soup.findAll("ul")
    return page_lists[-1]  # last list contains file names and links


def get_download_link(base_url, tag):
    download_link = urllib.parse.urljoin(base_url, tag["href"])
    check_url_status(download_link)
    filename = os.path.basename(download_link)
    return {filename: download_link}


def get_filepaths(soup, base_url):
    file_list = get_file_list(soup)
    texts = {}
    for element in file_list:
        tag = get_tag(element)
        is_tag_present = tag != -1
        if is_tag_present:
            texts.update(get_download_link(base_url, tag))
    return texts


def filter_model_files(userdefined_files, all_model_files):
    err_msg = "Filename {fname} not found in the model files hosted."
    filtered_model_files = {}
    if userdefined_files is not None:
        for fname in userdefined_files:
            if fname not in all_model_files:
                raise ValueError(err_msg.format(fname=fname))
            else:
                filtered_model_files[fname] = all_model_files[fname]
    return filtered_model_files


def create_and_get_download_dir(parent_dir, model_id):
    model_dir = os.path.join(parent_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_to_disk(all_model_files, model_id, directory, files_to_save=None):

    if files_to_save is not None:
        all_model_files = filter_model_files(files_to_save, all_model_files)

    model_dir = create_and_get_download_dir(directory, model_id)
    for filename in all_model_files:
        url_to_download = all_model_files[filename]
        temp_file_manager = partial(
            tempfile.NamedTemporaryFile, mode="wb", dir=directory, delete=False
        )
        cache_path = os.path.join(model_dir, filename)
        if os.path.exists(cache_path):
            continue
        with temp_file_manager() as temp_file:
            http_get(
                url_to_download,
                temp_file,
            )
        os.replace(temp_file.name, cache_path)
    cleanup_dir(directory)


def cleanup_dir(directory):
    new_filepaths = glob.glob(directory + "/*")
    for new_fp in new_filepaths:
        basename = os.path.basename(new_fp)
        is_temp_file = basename.startswith("tmp") and "." not in basename
        if is_temp_file:
            os.remove(new_fp)


def get_model_export_dir(base_url, download_dir):
    model_id = extract_model_id(base_url)
    if "/" in model_id:
        model_id = os.path.basename(model_id)
    model_dir = os.path.join(download_dir, model_id)
    if os.path.exists(model_dir):
        msg = f"Directory {model_dir} already exists. Files will be overwritten."
        warnings.warn(msg)
    return model_dir


def fetch_model(model_url, download_dir, files_to_save=None, upload_to_kaggle=False):
    corrected_model_url = get_correct_url(model_url)
    response = requests.get(corrected_model_url)
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    download_urls = get_filepaths(soup, corrected_model_url)
    print(
        Back.CYAN
        + Fore.BLACK
        + Style.BRIGHT
        + "Downloading files if necessary..."
        + Style.RESET_ALL
    )
    model_id = extract_model_id(corrected_model_url)

    save_to_disk(
        download_urls,
        files_to_save=files_to_save,
        model_id=model_id,
        directory=download_dir,
    )

    if upload_to_kaggle:

        api = KaggleApi()
        api.authenticate()

        if "/" in model_id:
            slug = model_id.replace("/", "-")
            assert model_id.count("/") == 1
            parent_name, model_name = model_id.split("/")
            metadata_dir = os.path.join(download_dir, parent_name)
        else:
            model_name = slug = model_id
            metadata_dir = os.path.join(download_dir, model_name)

        slug = re.sub("[^A-Za-z0-9]", "-", slug)
        dataset_name = f"HuggingFace {model_name}"
        if len(dataset_name) > 50:
            raise ValueError
        username = api.config_values["username"]
        metadata = {
            "title": dataset_name,
            "id": f"{username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }

        metadata_fp = os.path.join(metadata_dir, "dataset-metadata.json")
        with open(metadata_fp, mode="w") as f:
            json.dump(metadata, f)

        api.dataset_create_new(
            folder=metadata_dir, convert_to_csv=False, dir_mode="tar"
        )

        print(
            Back.CYAN
            + Fore.BLACK
            + "****** Folder uploaded successfully ******"
            + Style.RESET_ALL
        )
        print(
            Back.YELLOW
            + Fore.BLACK
            + f"URL (may take time to load): https://kaggle.com/{username}/{slug}"
            + Style.RESET_ALL
        )
