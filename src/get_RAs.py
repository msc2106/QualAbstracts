import pandas as pd
import requests
import numpy as np
from time import sleep, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import ceil
from os import listdir
import logging

logging.basicConfig(filename=f'./log/{datetime.now().strftime("%y%m%d%H%M%S_get_RA")}.txt', level=logging.INFO)

BATCHES = 10
TIMEOUT = 2
WRITE_CHUNK = 20
DOIRA_SEARCH = lambda doi: f'https://doi.org/doiRA/{doi}'
CR_SEARCH = lambda doi: f'https://api.crossref.org/works/{doi}/agency'
CR_HEADERS = {'User-Agent':r"QualAbstracts/0.1 (https://github.com/msc2106/QualAbstracts; mailto:mark.simon.cohen@gmail.com)"}

def get_crra(doi):
    # logging.info(f'{doi} sent {datetime.now()}')
    r = requests.get(CR_SEARCH(doi), headers=CR_HEADERS, timeout=TIMEOUT)    
    req_limit = int(r.headers['x-rate-limit-limit'])
    limit_interval = float(r.headers['x-rate-limit-interval'][:-1])
    # logging.info(f'{doi} received {datetime.now()}, waiting {limit_interval/req_limit*BATCHES}s')
    sleep(limit_interval/req_limit*BATCHES)
    try:
        ra = r.json()['message']['agency']['id']
    except:
        ra = np.NaN
    return ra


def get_doira_batch(doi_str, start, end):
    sleep(1)
    r = requests.get(DOIRA_SEARCH(doi_str), timeout=TIMEOUT)
    ra_list  = list(entry['RA'] if 'RA' in entry.keys() else np.NaN for entry in r.json())
    return start, end, ra_list


def setup_temp_output(batch_num):
    output_file = f'./data/RAs/batch{batch_num:02}.csv'
    with open(output_file, mode='w') as f:
        f.write("doi,ra\n")
    return output_file


def process_logging(doi_list, ra_list, batch_num, i, output_file):
    if i > 0 and i % WRITE_CHUNK == 0:
        with open(output_file, mode='a') as f:
            f.writelines([f'"{doi_list[i-WRITE_CHUNK+j]}","{ra_list[i-WRITE_CHUNK+j]}"\n' for j in range(WRITE_CHUNK)])
    if i%100 == 0:
        logging.info(f'{datetime.now()} Batch {batch_num}: {i}')


def make_cr_batch_processor(doi_list, start, end, batch_num):
    def batch_processor():
        output_file = setup_temp_output(batch_num)
        ra_list = []
        for i in range(end-start):
            ra_list.append(get_crra(doi_list[i]))
            process_logging(doi_list, ra_list, batch_num, i, output_file)
        return start, end, ra_list
    return batch_processor


def make_doira_batch_processor(doi_list, start, end, batch_num):
    sub_batch_size = 200
    def batch_processor():
        output_file = setup_temp_output(batch_num)
        ra_list=[]
        for sub_batch_start in range(0, len(doi_list), sub_batch_size):
            doi_str = ''
            sub_batch_end = min(sub_batch_start + sub_batch_size, len(doi_list))
            for doi in doi_list[sub_batch_start:sub_batch_end]:
                if doi_str != '':
                    doi_str += ','
                doi_str += doi
            r = requests.get(DOIRA_SEARCH(doi_str))
            ra_list.extend([entry['RA'] if 'RA' in entry.keys() else np.NaN for entry in r.json()])
            if len(ra_list) != sub_batch_end:
                print(len(ra_list), sub_batch_start, sub_batch_end)
                logging.info(doi_list[sub_batch_start:sub_batch_end])
                logging.info(r.json())
                raise RuntimeError("see log")
            with open(output_file, mode='a') as f:
                f.writelines([f'"{doi_list[sub_batch_start+i]}","{ra_list[sub_batch_start+i]}"\n' for i in range(sub_batch_end-sub_batch_start)])
            if len(ra_list)%1000 == 0:
                logging.info(f'{datetime.now()} Batch {batch_num}: {len(ra_list)}')
        return start, end, ra_list
    return batch_processor


def batch_iterator(df):
    n = df.shape[0]
    batch_size = ceil(n / BATCHES)
    for batch_num, start in enumerate(range(0, BATCHES*batch_size, batch_size)):
        end = min(start+batch_size, n)
        doi_list = list(df.doi.iloc[start:end])
        if batch_num % 2 == 0:
            # yield make_cr_batch_processor(doi_list, start, end, batch_num)
            yield make_doira_batch_processor(doi_list, start, end, batch_num)
        else:
            yield make_doira_batch_processor(doi_list, start, end, batch_num)



def read_prev():
    ra_files = listdir('./data/RAs')
    ra_dfs = []
    for file in ra_files:
        if file[-4:] == '.csv':
            ra_dfs.append(pd.read_csv(f"./data/RAs/{file}"))
    if len(ra_dfs) == 0:
        return pd.DataFrame({'doi':[], 'ra':[]})
    else:
        prev_ra = pd.concat(ra_dfs)
        prev_ra.drop_duplicates(subset=['doi'],inplace=True)
        prev_ra.to_csv('./data/RAs/all_prev.csv', index=False)
        return prev_ra



def runner(df):
    print('to be processed', df.shape[0])
    processed = 0
    threads = []
    with ThreadPoolExecutor() as executor:
        logging.info("start submitting threads")
        for batch in batch_iterator(df):
            threads.append(executor.submit(batch))
        for task in as_completed(threads):
            start, end, ra_list = task.result()
            df.ra.iloc[start:end] = ra_list
            processed += end-start
            logging.info(f'*************processed {processed} ***************')
    return df


def main():
    logging.info('loading DOIs')
    known_ra = read_prev()
    known_count = known_ra.shape[0]
    print(known_ra.ra.value_counts(dropna=False))
    doi_df = pd.read_csv('./data/all_articles_notext.csv', usecols=['doi'])
    # records without DOIs are useless
    doi_df.dropna(subset=['doi'], inplace=True)
    # remove duplicated DOIs
    doi_df.drop_duplicates(subset=['doi'], inplace=True)
    doi_df = doi_df[~doi_df.doi.str.contains('#')]
    print(f'Left to get: {doi_df.shape[0]-known_count}')
    doi_df = doi_df.merge(known_ra, how='left', on='doi')
    print(doi_df.head())
    doi_df = pd.concat([
        doi_df[doi_df.doi.isin(known_ra.doi)],
        runner(doi_df[~doi_df.doi.isin(known_ra.doi)])
    ])
    doi_df.to_csv('./data/doi_ra.csv', index=False)


if __name__ == '__main__':
    main()
