import torch
import create_batch
import model
import pandas
from torch.utils import data as torch_data

NUM_LAYERS = 40
MAX_SAMPLE = 6000
NUM_HIDDEN = 40
PREDICTION = [5, 10, 20, 30, 40]  # a list containing periods required to predict;
NUM_OUTPUT = len(PREDICTION)
NUM_BATCHES = 100
EPOCHS = 100
TRAIN_TEST_SPLIT = 0.9
VALIDATION_THRESHOLD = 0.4
loss_fn = torch.nn.MSELoss()


# when working on regression tasks, using MSE is required; (cross entropy is mainly used for classification)


def train(model, train_dataloader, valid_dataloader, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for iterations in range(EPOCHS):
        total_loss = 0
        prediction, actual = (None, None)

        # should never forget this step!!!!!! or gradiant will accumulate, leading to error!!!
        optimizer.zero_grad()

        for index, sample in enumerate(train_dataloader):
            device_sample, device_target = sample[0].to(device), sample[1].to(device)
            result = lstm(device_sample)
            prediction = result[0]
            actual = device_target[0]
            loss = loss_fn(result, device_target)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        average_loss = total_loss / (index * NUM_BATCHES)
        print("epoch {}: average training loss is ({})".format(iterations, average_loss))
        validate(model, valid_dataloader, device)
        # print("prediction: {}; actual: {}".format(prediction, actual))


def validate(model, valid_dataloader, device):
    """
    if the predicted price is within a certain range, will treat the prediction as correct.
    threshold is a hyperparameter set at beginning of file.

    :param model:
    :param valid_dataloader:
    :param device:
    :return:
    """
    model.eval()
    with torch.no_grad():
        correct_prediction = [0] * NUM_OUTPUT
        total_predictions = 0
        total_loss = 0
        for index, sample in enumerate(valid_dataloader):
            device_sample, device_target = sample[0].to(device), sample[1].to(device)
            result = lstm(device_sample)
            prediction = result
            actual = device_target[0]
            total_predictions += 1
            for period in range(len(prediction)):
                actual_low = actual[period] * (1 - VALIDATION_THRESHOLD)
                actual_high = actual[period] * (1 + VALIDATION_THRESHOLD)
                if actual_low <= prediction[period] <= actual_high:
                    correct_prediction[period] += 1
            loss = loss_fn(prediction, actual)
            total_loss += loss
        # print_statement should be modified;
        average_loss = total_loss / index
        print("average validation loss: " + str(average_loss.item()))
        print("number of in-threshold predictions: " + str(correct_prediction))
        print("number of total predictions: " + str(total_predictions) + "\n")


if __name__ == "__main__":
    data = pandas.read_csv("../data/spy1001-2201.csv")
    train_dataloader, valid_dataloader, feature_size = \
        create_batch.create_dataloader(data, NUM_LAYERS,
                                       MAX_SAMPLE, NUM_BATCHES,
                                       PREDICTION, TRAIN_TEST_SPLIT)
    print("data preparation ready")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = model.IntegratedModel(NUM_LAYERS, feature_size, NUM_HIDDEN, NUM_OUTPUT)
    train(lstm, train_dataloader, valid_dataloader, device)

