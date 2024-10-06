import os
import zipfile
import warnings
import argparse
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def save_in_zip(path_to_output: str, path_to_files: str = None, limit: int = 20 * (1024 ** 3)):
  """
  Zips the content of the folder `path_to_files` into zip files,
  each not exceeding the specified `limit` size.

  Args:
    path_to_output (str): The directory where the zip files will be saved.
    path_to_files (str): The directory containing files to zip.
    limit (int): Maximum size (in bytes) for each zip file.

  Raises:
    ValueError: If `path_to_files` does not exist or is not a directory.
  """

  if not os.path.isdir(path_to_files):
    raise ValueError(f"The path {path_to_files} does not exist")
  os.makedirs(path_to_output, exist_ok=True)

  # Collect files and their sizes
  files_to_zip: List[Tuple[str, int]] = []
  total_size = 0  # For progress bar total
  for root, _, files in os.walk(path_to_files):
    for file in files:
      filepath = os.path.join(root, file)
      filesize = os.path.getsize(filepath)
      if filesize > limit:
        warnings.warn(f"File {filepath} is larger than the limit ({limit} bytes) and will be skipped")
      else:
        files_to_zip.append((filepath, filesize))
        total_size += filesize

  files_to_zip.sort(key=lambda x: x[1], reverse=True)
  
  batches = []
  current_batch = []
  current_batch_size = 0

  for filepath, filesize in tqdm(files_to_zip, desc="Batching files"):
    if current_batch_size + filesize > limit:
      if current_batch:
        batches.append(current_batch)
      current_batch = [(filepath, filesize)]
      current_batch_size = filesize
    else:
      current_batch.append((filepath, filesize))
      current_batch_size += filesize

  # Add the last batch if not empty
  if current_batch:
    batches.append(current_batch)
  
  zip_files_created = []
  pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Zipping files")
  
  def zip_batch(batch_index, batch_files):
    zip_filename = os.path.join(path_to_output, f"files_{batch_index}.zip")
    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
      for filepath, filesize in batch_files:
        arcname = os.path.relpath(filepath, path_to_files)
        zipf.write(filepath, arcname=arcname)
        pbar.update(filesize)
    return zip_filename
  
  max_workers = min(16, len(batches))
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks
    future_to_batch = {
      executor.submit(zip_batch, index + 1, batch): index
      for index, batch in enumerate(batches)
    }

    # Collect results
    for future in as_completed(future_to_batch):
      zip_filename = future.result()
      zip_files_created.append(zip_filename)

  pbar.close()
  return zip_files_created

def parse_size(size_str):
  units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
  size_str = size_str.upper().strip()
  for unit in units:
    if size_str.endswith(unit):
      return int(float(size_str.rstrip(unit)) * units[unit])
  return int(size_str)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Zip the content of a folder into multiple zip files, each not exceeding a specified size limit, and upload them to a Hugging Face repository.")
  parser.add_argument("path_to_output", help="The directory where the zip files will be saved.")
  parser.add_argument("path_to_files", help="The directory containing files to zip.")
  parser.add_argument("--limit", type=parse_size, default="20GB", help="Maximum size for each zip file (e.g., 100MB, 2GB).")

  args = parser.parse_args()

  zip_files_created = save_in_zip(path_to_output=args.path_to_output, path_to_files=args.path_to_files, limit=args.limit)