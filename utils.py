import pickle
import datetime
from typing import List
import pandas as pd
from abc import ABC, abstractmethod

class CabaiPredictor(ABC):
    def __init__(self, model_name) -> None:
        """
        Loads model from file
        """
        with open(f"../models/{model_name}.pckl", "rb",) as fin:
            try:
                self.model = pickle.load(fin)
            except (OSError, FileNotFoundError, TypeError):
                print("Wrong path/model not available")
                exit(-1)

    def calculate_next_date(self, prev_date):
        """
        Calculate next date
        date format = yyyy-mm-dd
        """
        self.next_date = datetime.datetime(
            *list(map(lambda x: int(x), prev_date.split("-")))
        ) + datetime.timedelta(
            days=1
        )

    def get_next_date(self, prev_date):
        try:
            return self.next_date.strftime("%y-%m-%d")
        except NameError:
            self.calculate_next_date(prev_date)

    @abstractmethod
    def predict(self, prev_date) -> List:
        pass

    @abstractmethod
    def preprocess_inputs(self, prev_date):
        pass

    @abstractmethod
    def postprocess_outputs(self, output_from_model) -> List:
        pass

class FBProphetPredictor(CabaiPredictor):
    def __init__(self,) -> None:
        """
        Load model from file models/fbprophet.pckl
        """
        super().__init__("fbprophet")
    
    def preprocess_inputs(self, prev_date):
        """
        Model takes in an input as a pandas dataframe having index 
        as the day to be predicted
        """
        self.calculate_next_date(prev_date)
        next_date_series = pd.DataFrame(
            {"ds": pd.date_range(start=self.next_date, end=self.next_date)}
        )
        return next_date_series

    def postprocess_outputs(self, output_from_model) -> List:
        """
        return the ythat in the list format
        """
        return output_from_model["yhat"].tolist()

    def predict(self, prev_date) -> List:
        next_date_series = self.preprocess_inputs(prev_date)
        pred = self.model.predict(next_date_series)
        pred = self.postprocess_outputs(pred)
        return pred