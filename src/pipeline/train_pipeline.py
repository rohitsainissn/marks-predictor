import os
import sys

from src.exception import CustomException

import pandas as pd
from src.logger import logging

from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Additional steps if needed
        
        return model



# if __name__ == "__main__":
#     training_pipeline = TrainingPipeline()  # Creating an instance of TrainingPipeline
#     trained_model = training_pipeline.run_pipeline()  # Calling run_pipeline() method
