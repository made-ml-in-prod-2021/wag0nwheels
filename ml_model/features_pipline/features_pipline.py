from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from ml_model.enities.features_params import FeatureParams


def del_features(data: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    res = np.array(data.isnull().sum()) / data.shape[0]
    res = [idx for idx, percent in enumerate(res) if percent >= params.pass_limit]
    res = [data.columns[idx] for idx in res]
    return data.drop(columns=res)


def make_target(data: pd.DataFrame, params: FeatureParams) -> Tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=[params.target_name]), data[params.target_name]


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    #return pd.DataFrame(transformer.transform(df).toarray())
    return pd.DataFrame(transformer.transform(df))

def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
        ]
    )
    return transformer
