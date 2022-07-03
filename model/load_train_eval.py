import torch
import create_batch
import model
import pandas

NUM_LAYERS = 20
MAX_SAMPLE = 6000
NUM_HIDDEN = 30
PREDICTION = [10, 15, 20, 30, 40]  # a list containing periods required to predict;
NUM_OUTPUT = len(PREDICTION)
NUM_BATCHES = 200
EPOCHS = 100
loss_fn = torch.nn.MSELoss()
# when working on regression tasks, using MSE is required; (cross entropy is mainly used for classification)


def train():
    data = pandas.read_csv("../data/spy1001-2201.csv")
    dataloader, feature_size = create_batch.create_dataloader(data, NUM_LAYERS,
                                                              NUM_LAYERS, NUM_BATCHES, PREDICTION)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = model.IntegratedModel(NUM_LAYERS, feature_size, NUM_HIDDEN, NUM_OUTPUT)
    lstm.to(device)
    lstm.train()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    for iterations in range(EPOCHS):
        total_loss = 0
        prediction, actual = (None, None)

        # should never forget this step!!!!!! or gradiant will accumulate, leading to error!!!
        optimizer.zero_grad()

        for index, sample in enumerate(dataloader):
            device_sample, device_target = sample[0].to(device), sample[1].to(device)
            result = lstm(device_sample)
            prediction = result[0]
            actual = device_target[0]
            loss = loss_fn(result, device_target)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        print("epoch {}: loss is ({})".format(iterations, total_loss))
        print("prediction: {}; actual: {}".format(prediction, actual))


train()