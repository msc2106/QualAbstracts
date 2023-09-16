from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import evaluate
import platform
import pandas as pd
import numpy as np
from os import mkdir, listdir
from time import strftime
from sys import argv
import re
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    if len(argv) != 4:
        print("syntax: python train.py checkpoint subset epochs \nUse . to indicate latest checkpoint in ./models or to use the full training dataset")
        exit()
    working_dir = '.'
    data_dir = working_dir + '/data'
    models_dir = working_dir + '/models'

    tmp_dir = 'model_temp'
    if 'model_temp' not in listdir():
        mkdir(tmp_dir)
    if argv[1] == '.':
        model_name = sorted(listdir(models_dir)).pop()
        print("Latest model:", model_name)
        checkpoint = f'{models_dir}/{model_name}'
    else:
        model_name=argv[1]
        checkpoint=model_name
        print("Checkpoint:", checkpoint)

    ###### Model to be fine-tuned #####
    ### longt5: alternative version in order of memory intensity
    # checkpoint = 'google/long-t5-local-base'
    # checkpoint = 'google/long-t5-tglobal-base'
    # checkpoint = 'google/long-t5-local-large'
    # checkpoint = 'google/long-t5-tglobal-large'
    #### continue training last model
    ##################################

    num_epochs = int(argv[3])
    subset_size = None if argv[2] == '.' else int(argv[2])
    test_subset_size = None if not subset_size else subset_size // 16

    # environ['PJRT_DEVICE'] = 'GPU' # this seems to cause a cudnn version error
    training_args = Seq2SeqTrainingArguments(
        tmp_dir,
        bf16=True, #cause of loss failing to decrease?
        bf16_full_eval=True,
        #evaluation_strategy = 'epoch',
        per_device_eval_batch_size = 1,
        predict_with_generate = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        gradient_checkpointing = True,
        num_train_epochs = num_epochs,
        optim = 'adafactor',
        save_total_limit=1 # NB AdamW checkpoints are very large
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    def process_df(df, n):
        processed = df.iloc[:n, :].copy()
        # processed.fulltext = processed.fulltext.apply(lambda s: 'summarize: ' + s)
        return processed

    def prepare_data(data):
        return tokenizer(text=data['fulltext'], max_length=16384, truncation=True, text_target=data['abstract'])

    rouge = evaluate.load('rouge')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}



    train_df = pd.read_csv(data_dir+'/train.csv.gz')
    # test_df =  pd.read_csv(data_dir+'/test.csv.gz')
    # renaming = {
    #     'abstract':'label',
    #     'fulltext':'text'
    # }
    # train_df.rename(columns=renaming, inplace=True)
    print(train_df.info())
    # print(test_df.info())

    # randomize and split off validation data
    n = train_df.shape[0]
    n_val = n//10
    n_train = n - n_val
    rng = np.random.default_rng()
    shuffled_idx = rng.permutation(n)
    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:]

    # convert to HF Dataset
    train_data = Dataset.from_pandas(process_df(train_df.iloc[train_idx], subset_size)).map(prepare_data, batched=True, batch_size=4)
    val_data = Dataset.from_pandas(process_df(train_df.iloc[val_idx], test_subset_size)).map(prepare_data, batched=True, batch_size=4)

    # config = AutoConfig.from_pretrained(checkpoint, max_length=400)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        max_length=512,
        do_sample=True,
        num_beams=2,
        device_map='auto'
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_data,
        eval_dataset = val_data,
        tokenizer = tokenizer,
        data_collator = collator,
        compute_metrics = compute_metrics
    )
    # print(model.config_class.to_dict())

    trainer.train()

    timestamp = strftime('%Y%m%d%H%M')
    if len(argv) == 1:
        model_save_name = get_name(model_name) + timestamp
    else:
        cleaned_name = model_name.replace('/','_').replace('-','_')
        model_save_name = f'{cleaned_name}_tuned_{timestamp}'
    model_save_dir = f'{models_dir}/{model_save_name}'
    mkdir(model_save_dir)
    trainer.save_model(model_save_dir)
    print('saved to:', model_save_dir)

    print(trainer.evaluate())


def get_name(checkpoint):
    return re.match(r'.+(?=2023)', checkpoint)[0]


if __name__ == '__main__':
    main()
