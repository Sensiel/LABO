import os
import zipfile
import argparse
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def unzip_files(zip_files: List[str], output_dir: str, delete_zip: bool = False):
  os.makedirs(output_dir, exist_ok=True)

  total_files = 0
  for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
      total_files += len(zip_ref.infolist())
  print(f"Found {len(zip_files)} zip files containing {total_files} files to unzip.")
  pbar = tqdm(total=total_files, desc="Unzipping files")
  def unzip_file(zip_file):
    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
      for member in zip_ref.infolist():
        try:
          zip_ref.extract(member, output_dir)
        except FileExistsError:
          # Ignore
          pass
        finally:
          pbar.update(1)
    if delete_zip:
      os.remove(zip_file)

  with ThreadPoolExecutor() as executor:
    list(executor.map(unzip_file, zip_files))
  print("All zip files have been unzipped.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download zip files from a Hugging Face repository or unzip zip files from a local path.")
  parser.add_argument("zip_dir", help="The local path containing zip files to unzip.")
  parser.add_argument("output_dir", help="The directory where files will be extracted.")
  parser.add_argument("--delete_zip", action="store_true", help="If set, delete the zip files after extraction.")
  args = parser.parse_args()
    
  if not os.path.isdir(args.zip_dir):
    raise ValueError(f"Local path '{args.zip_dir}' is not a directory.")
  zip_files = [os.path.join(args.zip_dir, f) for f in os.listdir(args.zip_dir) if f.endswith('.zip')]
  if not zip_files:
    raise ValueError(f"No zip files found in local path '{args.zip_dir}'.")

  unzip_files(zip_files, output_dir=args.output_dir, delete_zip=args.delete_zip)
