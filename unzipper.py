import os
import zipfile
import argparse
from typing import List
from tqdm import tqdm
from huggingface_hub import HfApi, get_full_repo_name
from huggingface_hub.utils import RepositoryNotFoundError
from concurrent.futures import ThreadPoolExecutor


def download_zip_files_from_repo(repo_id: str, token: str, zip_dir: str):
  os.makedirs(zip_dir, exist_ok=True)
  download_command = [
  'huggingface-cli',
  'download',
  repo_id,
  '--repo-type',
  'dataset',
  '--local-dir',
  zip_dir,
  token
  ]
  subprocess.run(download_command, check=True)

def unzip_files(zip_files: List[str], output_dir: str, delete_zip: bool = False):
  os.makedirs(output_dir, exist_ok=True)

  total_files = 0
  for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
      total_files += len(zip_ref.infolist())
  print(f"Found {len(zip_files)} zip files containing {total_files} files to unzip.")

  def unzip_file(zip_file):
    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
      zip_ref.extractall(output_dir)
    if delete_zip:
      os.remove(zip_file)

  with ThreadPoolExecutor() as executor:
    executor.map(unzip_file, zip_files)
  print("All zip files have been unzipped.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download zip files from a Hugging Face repository or unzip zip files from a local path.")
  parser.add_argument("--repo_id", help="The Hugging Face repository ID (e.g., 'username/repo_name').")
  parser.add_argument("zip_dir", help="The local path containing zip files to unzip.")
  parser.add_argument("output_dir", help="The directory where files will be extracted.")
  parser.add_argument("--token", default=None, help="The Hugging Face access token. If not provided, will use the 'HF_TOKEN' environment variable.")
  parser.add_argument("--delete_zip", action="store_true", help="If set, delete the zip files after extraction.")
  args = parser.parse_args()

  token = args.token or os.getenv("HF_TOKEN")

  if args.repo_id:
    if not token:
      raise ValueError("A Hugging Face access token must be provided via the '--token' argument or the 'HF_TOKEN' environment variable when using '--repo_id'.")
    download_zip_files_from_repo(repo_id=args.repo_id, token=token, zip_dir=args.zip_dir)
    
  if not os.path.isdir(args.zip_dir):
    raise ValueError(f"Local path '{args.zip_dir}' is not a directory.")
  zip_files = [os.path.join(args.zip_dir, f) for f in os.listdir(args.zip_dir) if f.endswith('.zip')]
  if not zip_files:
    raise ValueError(f"No zip files found in local path '{args.zip_dir}'.")

  unzip_files(zip_files, output_dir=args.output_dir, delete_zip=args.delete_zip)
