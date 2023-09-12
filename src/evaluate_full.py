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
    if len(argv) == 1:
        model_name = sorted(listdir(models_dir)).pop()
        print('Evaluating newest model:', model_name)
    else:
        model_name = argv[1]
        print('Evaluating', model_name)
    if model_name in listdir(f'{working_dir}/evaluation'):
        raise FileExistsError('Target evaluation folder already exists. Remove before running')
    output_dir = f'{working_dir}/evaluation/{model_name}'
    mkdir(output_dir)

    checkpoint = f'{models_dir}/{model_name}'

    test_df = pd.read_csv(working_dir + '/data/test.csv.gz')
    test_df = test_df
    test_data = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, do_sample=True, num_beams=2, device_map='auto')
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device_map='auto')

    def gen(batch):
        return {'prediction': [item['summary_text'] for item in pipe(batch['fulltext'])]}    
    predictions = test_data.map(gen, batched=True, batch_size=4)

    rouge = evaluate.load("rouge")
    eval = rouge.compute(predictions=predictions['prediction'], references=predictions['abstract'])
    print(eval)
    with open(f'{output_dir}/rouge_score.txt', mode='w') as f:
        f.write(eval)
    for i, example in predictions:
        with open(f'{output_dir}/{i}.txt', mode='w') as f:
            f.write(f'ABSTRACT: \n{example["abstract"]} \n \nPREDICTION: \n{example["prediction"]} \n \nFULL TEXT: \n {example["fulltext"]}')


if __name__ == '__main__':
    main()

