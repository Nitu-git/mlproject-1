import os
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

num_cols = ["reading_score", "writing_score"]
cat_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts', "data_preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
           

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
            ])

            logging.info("num_pipeline created successfully")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical pipeline created successfully")

            logging.info(f"Numerical cols : {num_cols}")
            logging.info(f"Categorical cols:, {cat_cols}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("categorical_pipeline", categorical_pipeline, cat_cols)
            ])
            logging.info("Preprocessor created successfully...returning it")
            return preprocessor
        
        except Exception as e:
            logging.info("Data transformation failed")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name= "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Transformation applied successfully!!")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessing object.")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception occurred during data trasnformation initiation...")
            raise CustomException(e, sys)


        

