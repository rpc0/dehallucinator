import argparse
from deh.dl.squad import download_squad_qa_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="SQUAD v2.0 QA Dataset download")
    parser.add_argument("--cache_folder", default="./data/qa_dl_cache/")
    parser.add_argument("--context_folder", default="./data/contexts/")
    parser.add_argument("--qas_file", default="./data/qas/squad_qas.tsv")
    args = parser.parse_args()

    download_squad_qa_dataset(args.cache_folder, args.context_folder, args.qas_file)
