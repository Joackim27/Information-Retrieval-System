# run this code to create our file structure, usable to other projects
import os
from pathlib import Path
import logging

# Fix logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Create the files needed for this project with OS-independent paths
list_of_files = [
    os.path.join("src", "__init__.py"),
    os.path.join("src", "helper.py"),
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    os.path.join("research", "trials.ipynb"),
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = filepath.parent, filepath.name

    if filedir != Path(""):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
