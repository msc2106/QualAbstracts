import pandas as pd
import requests
from dotenv import load_dotenv
from os import getenv, mkdir, scandir, getcwd
import time

if load_dotenv():
    print("Loaded environment variables")
else:
    raise RuntimeError("Failed to load environemnt variables")
CORE_API_KEY = getenv('CORE_API_KEY')

def dir_exists(loc:str, dirname:str):
    with scandir(loc) as d:
        for entry in d:
            if entry.is_dir() and entry.name == dirname:
                return True
    return False


def try_query(url, headers, query):
    '''
    Returns:
    - results in this batch
    - total number of hits
    - the results list
    - the scroll ID
    '''
    delay = 5
    
    while True:
        try:
            res = requests.post(url, json=query, headers=headers)
            contents = res.json()
            remaining = int(res.headers['X-RateLimit-Remaining'])
            total_hits = contents['totalHits']
            scroll_id = contents['scrollId']
            results = contents['results']
        except Exception as e:
            print(f"{time.asctime(time.localtime(time.time()))} {type(e)}: {e}; status code: {res.status_code}")
            if res.status_code == 429:
                time.sleep(delay)
                delay *= 2
            continue
        break
    print(time.asctime(time.localtime(time.time())), "Remaining limit:", remaining)
    if remaining < 5:
        time.sleep(30 * (5-remaining))
    return len(results), total_hits, results, scroll_id


def start_params(data_dir):
    contents = scandir(data_dir)
    existing_records = 0
    file_count = 0
    for file in filter(lambda it: it.name.startswith('all_articles'), contents):
        file_count += 1
        df = pd.read_csv(f'{data_dir}/{file.name}')
        existing_records += df.shape[0]
        print(existing_records, "records")
    return f"{data_dir}/all_articles_{file_count}.csv", existing_records


def make_df(results_list) -> pd.DataFrame:
    data = [
        {
            'doi': entry['doi'],
            'title': entry['title'],
            'fulltext': entry['fullText'],
            'abstract': entry['abstract'],
            'issn': str(entry['journals'][0]['identifiers'])[1:-1] if len(entry['journals'])>0 else None,
            'subjects': entry['fieldOfStudy']
        }
        for entry in results_list 
            if {'doi', 'title', 'fullText', 'abstract', 'journals', 'fieldOfStudy'} <= entry.keys() 
                and type(entry['journals'])==list
    ]
    return pd.DataFrame(data)

def main():
    cwd = getcwd()
    if cwd.endswith('src'):
        rel_proj_dir = '..'
    elif cwd.endswith('QualAbstracts'):
        rel_proj_dir = '.'
    else:
        raise RuntimeError('Cannot determine relative directory')
    
    if not dir_exists(rel_proj_dir, 'data'):
        mkdir(f'{rel_proj_dir}/data')
    # save_file, offset = start_params(f'{rel_proj_dir}/data')
    save_file = f'{rel_proj_dir}/data/all_articles.csv'
    offset = 0
    
    url = 'https://api.core.ac.uk/v3/search/works'
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    query_template = {
        'q': '_exists_:acceptedDate _exists_:fullText _exists_:abstract',
        'exclude':['references'],
        'limit': 3000
    }
    
    count = offset
    first_query = {
        # 'offset': offset,
        'scroll': True,
        **query_template
    }
    result_size, total_hits, results, scroll_id = try_query(url, headers, first_query)
    # print(save_file, result_size, total_hits, scroll_id)
    # exit()
    
    if result_size == 0:
        raise RuntimeError("No results")
    count = result_size
    
    df = make_df(results)
    df.to_csv(save_file, index=False)
    print(f'{result_size} downloaded | {df.shape[0]} recorded | {count} of {total_hits}')
    
    while True:
        query = {
            'scrollId': scroll_id,
            **query_template
        }
        result_size, _, results, scroll_id = try_query(url, headers, query) 

        if result_size == 0:
            break
        count += result_size

        df = make_df(results)
        df.to_csv(save_file, index=False, mode='a', header=False)

        print(f'{result_size} downloaded | {df.shape[0]} recorded | {count} of {total_hits}')


if __name__ == '__main__':
    main()
