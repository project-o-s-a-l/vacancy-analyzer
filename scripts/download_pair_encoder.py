"""Explicit hash-pinned weight download; never called by serving."""

import argparse
from pathlib import Path

from jobs.cross_encoder import download_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    download_model(parser.parse_args().directory)
