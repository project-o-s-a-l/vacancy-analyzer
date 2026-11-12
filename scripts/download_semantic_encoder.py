"""Explicit hash-pinned encoder download; runtime never downloads models."""

import argparse

from jobs.semantic_encoder import ENCODERS, download_encoder, encoder_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    download_encoder(args.directory, args.model)
    print(f"Verified {args.model}: {encoder_id(args.model)}")


if __name__ == "__main__":
    main()
