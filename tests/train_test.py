import json
import os

from iris.train import Model

"""
Main function:

    Test the model from train.py and return results.

    1. Load the model from train.py.
    2. Load the iris dataset from a json file.
    3. Test the model on the dataset.
    4. Return the results.

---

test_main function:

    Test the main function.

    1. Run the main function.
    2. Compare the results with the expected results based on the output of the inference.py file.
    
"""

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