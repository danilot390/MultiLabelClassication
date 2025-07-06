from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd 
import numpy as np

class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Initialize the base model.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the training data.
            **kwargs: Additional keyword arguments for model training.
        Returns:
            Any: Trained model or relevant output.
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Predict using the model.
        
        Parameters:
            df(pd.DataFrame): Features to predict.
        Returns:
            np.ndarray: Predicted values.
        """
        pass

    @abstractmethod
    def data_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data for model training or prediction.
        
        Parameters:
            df (pd.DataFrame): DataFrame to transform.
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        pass

    def build(self, values: Dict[str, Any]={}) -> 'BaseModel':
        """
        Update the model's internal attributes using a dictionary of values.
        
        Parameters:
            values (Dict[str, Any]): Dictionary of values to build the model.
        Returns:
            BaseModel: Instance of the model with updated attributes.
        """
        if hasattr(self, 'defaults'):
            self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self