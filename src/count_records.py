import pandas as pd
from datetime import datetime
import os
from functools import reduce

def main():
    print('Start: ', datetime.now())
    data_files = os.listdir('./data')
    record_count = {}
    duplicates = {}
    DOIs = {}
    for file in data_files:
        print(file)
        record_count[file] = 0
        duplicates[file] = {'internal': 0}
        DOIs[file] = set()
        for chunk in pd.read_csv('./data/'+file, chunksize=10_000):
            new_records = chunk.shape[0]
            record_count[file] += new_records
            doi_set = set(chunk.doi)
            new_dups = new_records - len(doi_set) + len(doi_set & DOIs[file])
            duplicates[file]['internal'] += new_dups
            DOIs[file].update(doi_set)
            print('*', end=' ', flush=True)
        print(datetime.now())
    
    for file, doi_set in DOIs.items():
        duplicates[file]['external'] = len(DOIs[file] & reduce(set.union, (DOIs[k] for k in DOIs.keys() if k != file), set()))
    
    for file in data_files:
        print(f'{file}: {record_count[file]} records | {duplicates[file]} duplicates')


if __name__ == '__main__':
    main()
