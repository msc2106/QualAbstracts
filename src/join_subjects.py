import pandas as pd
import numpy as np
import gzip
import json
from datetime import datetime
from os import listdir
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
# logging.basicConfig(level=logging.DEBUG)

def main():
    doi_ra = pd.read_csv('./data/doi_ra.csv')
    global DOIs
    # DOIs = set(doi_ra[doi_ra.ra == 'Crossref'].doi)
    DOIs = doi_ra[doi_ra.ra == 'Crossref'].doi
    cr_data_dir = './data/crossref_data_2023_04'
    file_list = listdir(cr_data_dir)
    file_count = len(list(file_list))
    df_list = []
    all_subjects = set()
    print(datetime.now(), "start processing")
    threads = []
    processed = 0
    start_time = datetime.now()
    with ThreadPoolExecutor() as exec:
        for file in map(lambda f: f'{cr_data_dir}/{f}', file_list):
            threads.append(exec.submit(reader, file))
        for task in as_completed(threads):
            df, subjects = task.result()
            df_list.append(df)
            all_subjects.update(subjects)
            processed += 1
            if processed % 100 == 0:
                elapsed = datetime.now()-start_time
                print(f'******Processed {processed} files, total elapsed {elapsed}, est. {elapsed/processed * (file_count-processed)} remaining')

    all_df = pd.concat(df_list)
    print(f'Collected {len(all_subjects)} for {(all_df.subjects.apply(len)>0).sum()} records')
    all_df.to_csv('./data/doi_subject_journal.csv')
    with open('./data/all_subjects.txt', mode='w') as f:
        for subject in all_subjects:
            f.write(f'{subject}\n')


def reader(file):
    subjects = set()
    with gzip.open(file) as f:
        item_list = json.load(f)['items']
    # logging.debug('1')
    df = pd.DataFrame([
        {
            'doi': item['DOI'],
            'subjects': item['subject'] if 'subject' in item.keys() else [],
            'journal_title': item['container-title'][0] if 'container-title' in item.keys() else np.NaN
        }
        for item in item_list #if item['doi'] in DOIs
    ])
    # logging.debug('2')
    df = df[df.doi.isin(DOIs)]
    for subject_list in df.subjects:
        subjects.update(subject_list)
    print(datetime.now(), "done", file)
    return df, subjects



if __name__ == '__main__':
    main()
