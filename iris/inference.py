import json
import os
import pickle

from .train import Model


def main():
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