import logging
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_model.data.make_dataset import read_data, split_train_val_data
from ml_model.model.model import train, predict, evaluate, serialize_model
from ml_model.features_pipline.features_pipline import (del_features,
                                                        make_target,
                                                        process_categorical_features,
                                                        build_transformer
                                                        )
from ml_model.enities.training_params import TrainingParams
from ml_model.enities.features_params import FeatureParams
from ml_model.enities.split_params import SplittingParams

logger = logging.getLogger("train_pipeline")


def train_pipeline(logger):

    logger.info(f'Begin our experiment')
    data = 'heart.csv'
    try:
        df = read_data(data)
    except FileNotFoundError:
        logger.info('file {} not found'.format(data))
        return
    logger.info('dataset {} loaded'.format(data))
    FeatureParams.pass_limit = 0.7
    df = del_features(df, FeatureParams)
    logger.info(f'sparse features deleted')
    x_train, x_val = split_train_val_data(df, SplittingParams)
    x_train, y_train = make_target(x_train, FeatureParams)
    x_val, y_val = make_target(x_val, FeatureParams)
    logger.info(f'dataset splited on train and val')
    FeatureParams.numerical_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg',
                                        'thalach', 'exang', 'oldpeak']
    FeatureParams.categorical_features = ['cp', 'slope', 'ca', 'thal']
    my_transform = build_transformer(FeatureParams)
    x_train = my_transform.fit_transform(x_train)
    x_val = my_transform.transform(x_val)

    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_val = scalar.transform(x_val)

    model = train(x_train, y_train, FeatureParams, TrainingParams)
    logger.info(f'the model is trained')
    predicts = predict(model, x_val)
    logger.info('evaluate on train {}'.format(evaluate(predict(model, x_train), y_train)))
    logger.info('evaluate on validation {}'.format(evaluate(predicts, y_val)))


if __name__ == '__main__':
    logging.basicConfig(filename="ml_model_log.log", level=logging.INFO)
    train_pipeline(logger)
