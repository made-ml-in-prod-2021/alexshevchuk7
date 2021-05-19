import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    sample = pd.read_csv("sample.csv")
    columns = columns = ['cp', 'trestbps', 'restecg', 'slope', 'ca', 'thal']
    for i in range(10):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in sample[columns].iloc[i].tolist()
        ]
        response = requests.get(
            "http://127.0.0.1:8000/predict/",
            json={"data": [request_data]},
        )
        print(response.status_code)
        print(response.json())