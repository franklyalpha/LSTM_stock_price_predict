import pandas
import torch
import data_preprocessing.normalization as normali
import torch.utils.data as data_p
import math

"""
All operations assume the required indices/features in the data are already acquired. 

The purpose of this file is to hold methods to convert pandas data into torch tensors
for feeding into model and train, as well as methods for loading data into batches. 
"""


def remove_unnamed_and_date(data):
    """
    every time pandas reads from csv file would create a new unnamed column for
    storing indexes. Thus all of them should be removed before processing
    :param data:
    :return:
    """
    data = data[data.columns.drop(list(data.filter(regex='Unnamed.*')))]
    data = data.drop("Date", 1)
    return data


def normalize_sequence(sequential_data, target):
    """

    :param sequential_data: subset of pandas dataframe
           target: a sequence of integers representing price to predict for assigned periods
    :return: normalized target, which might have value greater than 1;
    """
    unnormalized_high, unnormalized_low = normali.normalize_price_average(sequential_data)
    normali.normalize_WR(sequential_data)
    normali.normalize_macd(sequential_data)
    normali.normalize_volume(sequential_data)
    for i in range(len(target)):
        normalized_target = normali.normalize_formula(target[i], unnormalized_high,
                                               unnormalized_low)
        target[i] = normalized_target


def generate_target(panda_data, curr_day, periods_list):
    """
    This is only an intuitive idea, might require further adjustments.

    :return:
    """
    future_prices = []
    # curr_price = panda_data["Close"][curr_day]
    for future_periods in periods_list:
        close_price = panda_data["Close"][curr_day + future_periods]
        future_prices.append(close_price)

    return future_prices


def create_dataloader(panda_data, sequence_length, max_samples,
                      batchsize, prediction_periods, train_test_factor=0.8):
    """
    realizing each sequential data piece has size: sequential_length * feature size;

    procedure: acquire "max_samples" many pieces of samples, each will be a torch tensor
    with size sequential length * feature size.
    They will all be appended into a new list, and be constructed as "StockDataset"(see description below)
    Then a dataloader will be created and returned.

    :return: a dataloader object for training iterations, and feature size
    """

    cleaned_data = remove_unnamed_and_date(panda_data)
    tensor_data = []
    target_data = []
    # ensure only "numeric data" is in panda_data, for converting to numpy
    a = len(cleaned_data) - max(prediction_periods) - 1
    for num_samples in range(min(max_samples,
                                 len(cleaned_data) - max(prediction_periods) - 1)):
        subset = cleaned_data.iloc[num_samples:(num_samples + sequence_length)].copy()

        target = generate_target(cleaned_data, num_samples, prediction_periods)

        normalize_sequence(subset, target)

        numpy_sample = subset.to_numpy()

        if numpy_sample.shape[0] < sequence_length:
            # handle cases that max_samples is larger than total number of samples can generate.
            break
        else:
            tensor_data.append(torch.tensor(numpy_sample, dtype=torch.float32))
            target_data.append(torch.tensor(target, dtype=torch.float32))
    dataset = StockDataset(tensor_data, target_data)
    train_size = math.ceil(len(dataset) * train_test_factor)
    split_dataset = data_p.random_split(dataset, [train_size,
                                                  len(dataset) - train_size])

    train_dataloader = data_p.DataLoader(split_dataset[0], batch_size=batchsize)
    valid_dataloader = data_p.DataLoader(split_dataset[1], batch_size=1)
    return train_dataloader, valid_dataloader, tensor_data[0].shape[1]


class StockDataset(data_p.Dataset):
    def __init__(self, data, target):
        """
        :param data: each single element from the "sequential data" has dimension sequence_length *
        feature_size.
        """
        super(StockDataset, self).__init__()
        self.sequential_data = data
        self.target = target

    def __len__(self):
        return len(self.sequential_data)

    def __getitem__(self, item):
        """
        :param item: the index to acquire the data
        :return:
        """
        return self.sequential_data[item], self.target[item]
