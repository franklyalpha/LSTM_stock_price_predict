import torch
import create_batch
import model
import pandas

NUM_LAYERS = 60
MAX_SAMPLE = 40
NUM_HIDDEN = 80
NUM_OUTPUT = 9
NUM_BATCHES = 10

data = pandas.read_csv("../data/spy1001-2201.csv")
dataloader, feature_size = create_batch.create_dataloader(data, NUM_LAYERS, NUM_LAYERS, NUM_BATCHES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm = model.IntegratedModel(NUM_LAYERS, feature_size, NUM_HIDDEN, NUM_OUTPUT)
lstm.to(device)
for index, sample in enumerate(dataloader):
    device_sample = sample.to(device)
    result = lstm(device_sample)



