import json
import os

from iris.train import Model


def main():
    m = Model()
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


    return records


def test_main():
    records = main()

    # Your assertion here
    assert records[0]['label'] == 'versicolor'

# {"dataset": "iris", "architecture": "KNN", "features": 0.96, "data": [1, 2, 3, 4], "label": "versicolor"}