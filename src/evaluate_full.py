from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import evaluate
from os import listdir, mkdir
import pandas as pd
import numpy as np
from sys import argv


def main():
    working_dir = '.'
    models_dir = working_dir + '/models'
    testing = len(argv) > 1 and argv[1] == '-test'
    write_testing = len(argv) > 1 and argv[1] == '-writetest'
    if testing:
        print('****Testing mode****')
    if len(argv) == 1 or ((testing or write_testing) and len(argv)==2):
        model_name = sorted(listdir(models_dir)).pop()
        print('Evaluating newest model:', model_name)
        checkpoint = f'{models_dir}/{model_name}'
    else:
        model_name = argv[-1]
        print('Evaluating', model_name)
        checkpoint = model_name
    if not testing:
        model_save_name = model_name.replace('/','_').replace('-','_')
        if model_save_name in listdir(f'{working_dir}/evaluation'):
            raise FileExistsError('Target evaluation folder already exists. Remove before running')
        output_dir = f'{working_dir}/evaluation/{model_save_name}'
        mkdir(output_dir)

    nrows = 12 if testing or write_testing else None
    test_df = pd.read_csv(working_dir + '/data/test.csv.gz', nrows=nrows)
    test_df = test_df
    test_data = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=16384, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, do_sample=True, num_beams=2, device_map='auto', trust_remote_code=True)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device_map='auto')

    def gen(batch):
        return {'prediction': [item['summary_text'] for item in pipe(batch['fulltext'], truncation=True)]}
    predictions = test_data.map(gen, batched=True, batch_size=4)

    rouge = evaluate.load("rouge")
    eval = rouge.compute(predictions=predictions['prediction'], references=predictions['abstract'])
    print(eval)
    if testing:
        print(f"\nABSTRACT (1st):\n{predictions[0]['abstract']}\n\nPREDICTION (1st):\n{predictions[0]['prediction']}")
    else:
        with open(f'{output_dir}/rouge_score.txt', mode='w') as f:
            for key, val in eval.items():
                f.write(f'{key}: {val}\n')
        for i, example in enumerate(predictions):
            with open(f'{output_dir}/{i}.txt', mode='w') as f:
                f.write(f'ABSTRACT: \n{example["abstract"]} \n \nPREDICTION: \n{example["prediction"]} \n \nFULL TEXT: \n {example["fulltext"]}')


if __name__ == '__main__':
    main()
