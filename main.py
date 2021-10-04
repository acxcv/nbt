# DO ALL NECESSARY IMPORTS
from nbt_model import FinalModel
from model_evaluation import evaluate_model
from model_training import train_model
from load_data import create_data_loader


def __main__():

    model = FinalModel()

    # Filepaths used for training and evaluating
    pickle_train_path = 'vector_frame_train'
    pickle_validation_path = 'vector_frame_validate'
    pickle_test_path = 'vector_frame_test'

    # Load data
    training_data = create_data_loader(pickle_train_path)
    validation_data = create_data_loader(pickle_validation_path)
    test_data = create_data_loader(pickle_test_path)
    dummy_training = create_data_loader('vector_frame_train_dummy')
    dummy_validate = create_data_loader('vector_frame_validate_dummy')
    dummy_test = create_data_loader('vector_frame_test_dummy')

    # Call model training
    # train_model(model, 3, training_data)

    # Call model evaluation
    evaluate_model(dummy_test,
                   'saved_models/saved_model_2021-09-15_19-43')


if __name__ == '__main__':
    __main__()
