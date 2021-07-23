import numpy as np
import pandas as pd

import dask.dataframe as dd
from dask.layers import DataFrameLayer


def test_dataframelayer_base():

    # Test that various dask.dataframe operations
    # will produce high-level-graphs comprising
    # only of DataFrameLayer-based Layer objects
    df = pd.DataFrame(
        {
            "a": np.arange(24),
            "b": ["dog", "cat"] * 12,
            "c": np.random.randint(10, size=24),
        }
    )
    ddf = dd.from_pandas(df, npartitions=3)
    ddf = ddf.set_index("c", shuffle="tasks")
    for key, layer in ddf.dask.layers.items():
        assert isinstance(layer, DataFrameLayer)

    # Include a final operation that is known
    # to be a "materialized" Layer
    tail = ddf.tail(compute=False)
    for key, layer in tail.dask.layers.items():
        assert isinstance(layer, DataFrameLayer)
