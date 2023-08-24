import pandas as pd
import numpy as np
import gzip
import json
from datetime import datetime
from os import listdir
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
# logging.basicConfig(level=logging.DEBUG)

cr_data_dir = './data/crossref_data_2023_04'
COLUMNS = ['DOI', 'subject', 'container-title', 'title', 'abstract', 'language']

def make_doi_list():
    doi_ra = pd.read_csv('./data/doi_ra.csv')
    # dois = set(doi_ra[doi_ra.ra == 'Crossref'].doi)
    dois = doi_ra[doi_ra.ra == 'Crossref'].doi.sort_values().to_numpy()
    print('make dois')
    return dois, set(dois)

DOIs, DOI_SET = make_doi_list()

def testing():
    return reader(f'{cr_data_dir}/0.json.gz')


def main():
    file_list = listdir(cr_data_dir)
    file_count = len(list(file_list))
    df_list = []
    print(datetime.now(), "start processing")
    threads = []
    processed = nonzero = 0
    start_time = datetime.now()
    with ThreadPoolExecutor() as exec:
        for file in map(lambda f: f'{cr_data_dir}/{f}', filter(lambda s: s[-3:]=='.gz',file_list)):
            threads.append(exec.submit(reader, file))
        for task in as_completed(threads):
            df = task.result()
            if df is not None:
                df_list.append(df)
                nonzero += 1
            processed += 1
            if nonzero > 0 and nonzero % 100 == 0:
                elapsed = datetime.now()-start_time
                print(f'******Processed {processed} files, total elapsed {elapsed}, est. {elapsed/processed * (file_count-processed)} remaining')
                df_list = [pd.concat(df_list)]

    all_df = pd.concat(df_list)
    all_df.to_csv('./data/cr_data.csv', index=False)


def doi_present(doi):
    loc = np.searchsorted(DOIs, doi)
    return not (loc == DOIs.shape[0] or DOIs[loc] != doi)

def reader(file):
    with gzip.open(file) as f:
        item_list = json.load(f)['items']
        # raw_json = json.load(f)
    # # logging.debug('1')
    df = pd.DataFrame([
        {
            'DOI': item['DOI'],
            'subjects': item['subject'] if 'subject' in item.keys() else [],
            'journal_title': item['container-title'][0] if 'container-title' in item.keys() else np.NaN,
            'title': item['title'] if 'title' in item.keys() else np.NaN,
            'abstract': item['abstract'] if 'abstract' in item.keys() else np.NaN,
            'lang': item['language'] if 'language' in item.keys() else np.NaN
        }
        for item in item_list if doi_present(item['DOI'])
    ])
    # df.columns=['DOI', 'subjects', 'journal_title', 'title', 'abstract', 'language']
    # logging.debug('2')
    # df.to_csv(f'{cr_data_dir}/df_parts/{file}.csv', index=False)
    
    # df = pd.json_normalize(
    #     item_list
    # )[COLUMNS]
    
    # df = df[df.DOI.isin(DOIs)]

    print(datetime.now(), "done", file, df.shape)
    return df if df.shape[0] > 0 else None



if __name__ == '__main__':
    main()
