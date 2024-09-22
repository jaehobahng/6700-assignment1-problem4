import json
import os
import pickle

from .train import Model


def main():
    """
    Main function to train the model from train.py, evaluate it on the iris dataset and save the results to a json file.

    1. Load the model from train.py and save it to a pickle file.
    2. Load the iris dataset from a json file.
    3. Train the model on the dataset.
    4. Evaluate the model on the dataset.
    5. Save the results to a json file.
    """

    m = Model()
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(m, model_file)

    with open ('./iris_data.json','r') as file:
        data = json.load(file)
    # os.environ["DATA"] = "../iris_data.json"
    # data = os.getenv("DATA")
    if not data:
        raise ValueError("No data provided")

    # data = json.loads(data)
    records = [
        {
            "dataset": m.dataset,
            "architecture": m.architecture,
            "features": m.eval,
            "data": record,
            "label": label,
        }
    for record, label in zip(data, m(data))]

    json.dump(records, open("out.json", "w"))

# set DATA=C:\path\to\data
# python main.py