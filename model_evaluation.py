from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from nbt_model import FinalModel
import torch
from tqdm import tqdm


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def evaluate_model(test_dl, model_path):

    # instantiate model, load last config, set eval mode
    model = FinalModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions, actuals = list(), list()

    loop = tqdm(test_dl, total=len(test_dl), leave=True)

    for batch in loop:
        # evaluate the model on the test set
        output = model(batch)
        output = [listt.detach().numpy() for listt in output]
        targets = [sample['binary_label'] for sample in batch]
        predictions.append(output)
        actuals.append(targets)
        # print(f'appended preds and actuals {i+1}/{len(test_dl)}')
    # predictions, actuals = vstack(predictions), vstack(actuals)
    # print('Predictions: ', predictions, '\nActuals: ', actuals)

    # calculate accuracy
    # TODO: either import sklearn or find workaround
    acc = accuracy_score(flatten(actuals), flatten(predictions))
    save_info = f'Model: {model_path}\nAccuracy: {acc}'
    with open(
            f'{model_path}_evaluation_dummy.log', 'w+', encoding='utf8') as logfile:
        logfile.write(save_info)
    return save_info
