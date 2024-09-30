import argparse
from deh.dl.squad import download_squad_qa_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="SQUAD v2.0 QA Dataset download")
    parser.add_argument("--cache_folder", default="./data/qa_dl_cache/")
    parser.add_argument("--context_folder", default="./data/contexts/")
    parser.add_argument("--qas_file", default="./data/qas/squad_qas.tsv")
    parser.add_argument("--limit_size", default=None)
    args = parser.parse_args()

    # Console print output:
    print("Downloading SQuAD v2.0 QA Dataset")
    print(f"cache_folder: {args.cache_folder}")
    print(f"context_folder: {args.context_folder}")
    print(f"qas_flie: {args.qas_file}")
    print(f"limit_size: {args.limit_size}")

    download_squad_qa_dataset(
        args.cache_folder, args.context_folder, args.qas_file, args.limit_size
    )
