from typing import List
import pandas as pd
import xml.etree.ElementTree as ET
import requests

BUCKET_URL = "https://tune-experiment-datasets.s3.us-west-2.amazonaws.com"

def get_classical_ml_datasets(biggest_first: bool = False) -> List[str]:
    xml = requests.get(BUCKET_URL).text
    root = ET.fromstring(xml)
    datasets = []
    for child in root:
        if "Contents" in child.tag:
            dataset = [None] * 2
            for child2 in child:
                if "Key" in child2.tag and "classical_ml/" in child2.text and ".parquet" in child2.text:
                    dataset[0] = child2.text
                if "Size" in child2.tag:
                    dataset[1] = int(child2.text)
            if dataset[0]:
                datasets.append(dataset)
    datasets.sort(key=lambda x: x[1], reverse=biggest_first)
    datasets = [f"{BUCKET_URL}/{x[0]}" for x in datasets]
    return datasets
