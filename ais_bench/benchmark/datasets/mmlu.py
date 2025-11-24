import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError

from .base import BaseDataset

logger = AISLogger()

@LOAD_DATASET.register_module()
class MMLUDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, **kwargs):
        path = get_data_path(path, local_mode=True)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            raw_data = []
            filename = osp.join(path, split, f'{name}_{split}.csv')
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader):
                    if len(row) != 6:
                        raise AISBenchDataContentError(
                            DSET_CODES.DATA_INVALID_STRUCTURE,
                            f"Row {row_idx} in {filename} has {len(row)} columns, expected 6"
                        )
                    raw_data.append({
                        'input': row[0],
                        'A': row[1],
                        'B': row[2],
                        'C': row[3],
                        'D': row[4],
                        'target': row[5],
                    })
            dataset[split] = Dataset.from_list(raw_data)
        return dataset