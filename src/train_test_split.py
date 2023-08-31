import pandas as pd
import numpy as np

def main():
    dataset = pd.read_csv('./data/abstracts_fulltext_cleaned_exact.csv.gz')
    n = dataset.shape[0]
    test_size = n//5
    train_size = n - test_size
    RNG = np.random.default_rng(100)
    permutation = RNG.permutation(n)
    train_idx = permutation[:train_size]
    test_idx = permutation[-test_size:]
    train_data = dataset.iloc[train_idx,:]
    test_data = dataset.iloc[test_idx,:]
    print("training data:", train_data.shape)
    print("testing data:", test_data.shape)
    assert not np.any(test_data.abstract.isin(train_data.abstract)), 'Duplicates'
    print('saving')
    train_data.to_csv('./data/train.csv.gz', index=False)
    test_data.to_csv('./data/test.csv.gz', index=False)
    print('validation')
    validate_test_data = pd.read_csv('./data/test.csv.gz')
    assert np.all(test_data.reset_index(drop=True) == validate_test_data), 'Validation failed'


if __name__ == '__main__':
    main()
