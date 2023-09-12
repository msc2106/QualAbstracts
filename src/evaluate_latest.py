from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import evaluate
from os import listdir
import pandas as pd
import numpy as np

working_dir = '.'
models_dir = working_dir + '/models'
latest_model = sorted(listdir(models_dir)).pop()
output_file = f'{working_dir}/evaluation/{latest_model}.txt'
finetuned = f'{models_dir}/{latest_model}'
sample_size = 256

print('Newest model:', latest_model)

test_df = pd.read_csv(working_dir + '/data/test.csv.gz')
first_record = test_df.iloc[0,:]
test_df = test_df.sample(sample_size)
test_data = Dataset.from_pandas(test_df)

def gen(batch):
    return {'prediction': [item['summary_text'] for item in pipe(batch['fulltext'])]}

tokenizer = AutoTokenizer.from_pretrained(finetuned)
model = AutoModelForSeq2SeqLM.from_pretrained(finetuned, do_sample=True, num_beams=2, device_map='auto')
pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device_map='auto')
first_pred = pipe(first_record['fulltext'])[0]['summary_text']
predictions = test_data.map(gen, batched=True, batch_size=4)

rouge = evaluate.load("rouge")
eval = rouge.compute(predictions=predictions['prediction'], references=predictions['abstract'])
print(eval)

# i = np.random.randint(sample_size)

output = f"""{eval}
ABSTRACT (1st)
{first_record['abstract']}
PREDICTION (1st)
{first_pred}
ABSTRACT (sample)
{predictions['abstract'][0]}
PREDICTION (sample)
{predictions['prediction'][0]}
"""

with open(output_file, mode='w') as f:
    f.write(output)

print(output)
