from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_model.enities.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_val_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame,
                                                                               pd.DataFrame]:
    '''
    Split data to train and val sample

    Parameters
    ---------------
    data: Pd.DataFrame
        Data to split
    params: float
        params to split

    Return
    ---------------
    Tuple of Train and Val pd.DataFrame 
    '''
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data

