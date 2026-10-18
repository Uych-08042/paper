import io

import numpy as np


def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        last_error = None
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except Exception as error:
                last_error = error

        # OTB contains annotations with comma and whitespace delimiters in the
        # same file. Normalize separators in memory without modifying data.
        if "," in delimiter and None in delimiter:
            try:
                with open(path, "r", encoding="utf-8-sig") as source:
                    normalized = source.read().replace(",", " ")
                return np.loadtxt(io.StringIO(normalized), dtype=dtype)
            except Exception as error:
                last_error = error

        raise Exception('Could not read file {}'.format(path)) from last_error
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    import pandas as pd

    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str
