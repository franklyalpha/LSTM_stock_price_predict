import torch
import create_batch as create_batch
import model
import pandas
from torch.utils import data as torch_data
import math

NUM_LAYERS = 50
MAX_SAMPLE = 20000
NUM_HIDDEN = 60
PREDICTION = [i for i in range(5, 61, 4)]  # a list containing periods required to predict;
NUM_OUTPUT = len(PREDICTION)
NUM_BATCHES = 200
EPOCHS = 100
TRAIN_TEST_SPLIT = 0.9
VALIDATION_THRESHOLD = 0.25

EARLY_STOP_THRESHOLD = 10
lr_milestones = [17, 40, 75]
lr_decay_gamma = 0.5

loss_fn = torch.nn.MSELoss()


# when working on regression tasks, using MSE is required; (cross entropy is mainly used for classification)


def train(model, train_dataloader, valid_dataloader, device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # learning rate scheduler:

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones, gamma=lr_decay_gamma)

    # early stopping:
    curr_best = model.state_dict()
    curr_lowest_val_accuracy = math.inf
    stop_count = 0

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
        lr_schedule.step()
        average_loss = total_loss / NUM_BATCHES
        print("epoch {}: average training loss is ({})".format(iterations, average_loss))

        # early stopping
        validation_accuracy = validate(model, valid_dataloader, device)
        model_saved = False
        if validation_accuracy < curr_lowest_val_accuracy:
            curr_lowest_val_accuracy = validation_accuracy
            curr_best = model.state_dict()
            stop_count = 0
            model_saved = True
        else:
            stop_count += 1
            if stop_count == EARLY_STOP_THRESHOLD:
                print("reached early stopping threshold, training stopped")
                break
        print("early_stopping_watcher: stop count: {}; model saved: {}\n".format(stop_count, model_saved))
        # print("prediction: {}; actual: {}".format(prediction, actual))
    # save model
    torch.save(curr_best, "../model/model_state.pt")


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
                # actual_low = actual[period] * (1 - VALIDATION_THRESHOLD)
                # actual_high = actual[period] * (1 + VALIDATION_THRESHOLD)
                actual_low = actual[period] - VALIDATION_THRESHOLD
                actual_high = actual[period] + VALIDATION_THRESHOLD
                if actual_low <= prediction[period] <= actual_high:
                    correct_prediction[period] += 1
            loss = loss_fn(prediction, actual)
            total_loss += loss
        # print_statement should be modified;
        accuracy = []
        for i in range(len(correct_prediction)):
            accuracy.append(round((correct_prediction[i] / total_predictions), 2))
        average_loss = total_loss / index
        print("average validation loss: " + str(average_loss.item()))
        print("number of in-threshold predictions: " + str(accuracy))
        print("number of total predictions: " + str(total_predictions))

    # calculate evaluation score, based on a weight measure emphasizing long term prediction
    return average_loss


# def load_and_predict()


if __name__ == "__main__":
    # if you want to perform continuous training, please use console
    # so that data will be loaded only once.

    """
    training comments: 
    by using the set hyperparameter, the training will achieve lowest loss at around epoch 15; 
    should consider lowering learning rate after this epoch. 
    
    """

    data = pandas.read_csv("../data/spy1001-2201.csv")
    # data = pandas.read_csv("data/spy1001-2201.csv") # use this line when using console
    train_dataloader, valid_dataloader, feature_size = \
        create_batch.create_dataloader(data, NUM_LAYERS,
                                       MAX_SAMPLE, NUM_BATCHES,
                                       PREDICTION, TRAIN_TEST_SPLIT)
    print("data preparation ready")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = model.IntegratedModel(NUM_LAYERS, feature_size, NUM_HIDDEN, NUM_OUTPUT)
    print("prediction period: " + str(PREDICTION))
    train(lstm, train_dataloader, valid_dataloader, device)

