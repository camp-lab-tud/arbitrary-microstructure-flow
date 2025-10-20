import os
import os.path as osp
import random
from urllib.parse import urlparse
import requests
import zipfile



user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
]

def download_data(url: str, save_dir: str) -> str:
    """
    Download data from URL.

    Args:
        url: URL of the data.
        save_dir: directory where data is stored.
    """

    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    zip_filename = get_zip_filename(url)
    zip_path = osp.join(save_dir, zip_filename)

    print(f'Downloading data from {url} ...')
    headers = {"User-Agent": random.choice(user_agents)} # add header to avoid being detected as bot
    response = requests.get(url, headers=headers)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print(f'Data downloaded to {zip_path}.')

    return zip_path


def unzip_data(zip_path: str, save_dir: str) -> str:
    """
    Extract data from zip file.

    Args:
        zip_path: path to the zip file.
        save_dir: directory where data is extracted to.
    """
    print(f'Extracting data from {zip_path}...')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # the zip file contains a single folder (with subfolders/files)
        namelist = zip_ref.namelist()
        folder_name = namelist[0].split('/')[0]
        zip_ref.extractall(save_dir)

    folder_path = osp.join(save_dir, folder_name)
    print(f'Data extracted to {folder_path}.')
    return folder_path


def is_url(s: str) -> bool:
    """Return True if string is a valid URL (http or https)."""
    try:
        parsed = urlparse(s.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def get_zip_filename(link: str) -> str:
    """
    Get the zip filename from a zenodo download link.

    Args:
        link: zenodo download link that is like 'https://zenodo.org/records/16940478/files/simulations.zip?download=1'
    
    Returns:
        out: zip filename, i.e., 'simulations.zip'.
    """
    out = link.split('/')[-1].split('?')[0]
    return out