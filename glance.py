import argparse
import os
from bltzr import SqlDataset, SqlDatasetConfig, Tokenizer

if __name__ == "__main__":

    env_db = os.environ.get('LLM_TRAIN_DB')
    env_user = os.environ.get('LLM_TRAIN_DB_USER')
    env_pass = os.environ.get('LLM_TRAIN_DB_PASS')
    env_host = os.environ.get('LLM_TRAIN_DB_HOST')
    if env_host is None:
        env_host = "127.0.0.1"

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', required=False, type=int, default=0, help="Start index of the dataset")
    parser.add_argument('-b', '--batch', required=False, type=int, default=100, help="Batch size for cache building")
    parser.add_argument("-n", "--count", type=int, default=5, required=False, help="Count of dataset items to fetch")
    parser.add_argument("-c", "--chunk", type=int, default=2048, required=False, help="Chunk size")
    parser.add_argument('-t', '--dataset', required=False, default='dataset', type=str, help="Dataset table name")
    parser.add_argument('-m', '--metadata', required=False, default=False, type=bool, help="Use metadata")
    parser.add_argument('--host', required=False, default=env_host, type=str, help="Database host")
    args = parser.parse_args()


    tokenizer = Tokenizer()
    data_config = SqlDatasetConfig(db_host=env_host, db_pass=env_pass, db_user=env_user, db_name=env_db, dataset_table=args.dataset, window_size=args.chunk, with_metadata=args.metadata, batch_size=args.batch)
    dataset = SqlDataset(data_config)
    print(f'Dataset len is {len(dataset)}')

    for i in range(args.count):
        if args.start + i < len(dataset):
            print(f'-{i}------')
            print(tokenizer.decode(dataset[args.start + i]['input_ids']))
            print(f'------{i}-')
        else:
            print(f'{args.start + i} index is greater than dataset length')
            break
