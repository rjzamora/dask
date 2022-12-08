from fsspec.core import get_fs_token_paths
from contextlib import ExitStack

import itertools
from dask.delayed import delayed

import dask
import dask.dataframe as dd
from dask.dataframe.io.utils import DataFrameIOFunction, _is_local_fs
from dask.utils import natural_sort_key, parse_bytes

import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

import pandas as pd
import cudf

from cudf.utils.ioutils import _open_remote_files
from cudf.io.parquet import _default_open_file_options

_BLOCKSIZE_DEFAULT = 64_000_000



class ReadParquet(DataFrameIOFunction):

    def __init__(self, engine, meta, map_options):
        self.engine = engine
        self.meta = meta
        self.map_options = map_options

    @property
    def columns(self):
        return self.meta.columns

    def project_columns(self, columns):
        if columns == self.columns:
            return self
        return ReadParquet(
            self.engine,
            self.meta[columns],
            self.index,
        )

    def __call__(self, part, **kwargs):
        return self.engine.make_partition(part, **self.map_options, **kwargs)


class Engine:

    @classmethod
    def gather_metadata(
        cls,
        urlpath,
        columns=None,
        index=None,
        filters=None,
        engine=None,
        blocksize="auto",
        chunksize=None,
        granularity="row-group",
        groups=None,
        storage_options=None,
        dataset_options=None,
        read_options=None,
    ):
        raise NotImplemetedError
    
    
    @classmethod
    def make_partitioning_plan(cls, fragments, partitioning_options):
        
        chunksize = partitioning_options["chunksize"]
        blocksize = partitioning_options["blocksize"]
        granularity = partitioning_options["granularity"]

        # Use fragment_info to calculate partitions
        def _partitioning_plan(metadata, size, size_field, granularity):

            if size_field not in ("num_rows", "total_byte_size", None):
                raise ValueError

            if size_field is None:
                partitions = list(range(len(metadata)))
            else:
                partitions = []
                shift, part = 0, 0
                for num_rows in metadata[size_field]:
                    if num_rows + shift > size:
                        shift = 0
                        part += 1
                    shift += num_rows
                    partitions.append(part)

            return metadata.groupby(partitions).agg({"path": list, "row_group": list})

        if chunksize:
            size_metric = chunksize
            size_field = "num_rows"
        elif blocksize:
            size_metric = blocksize
            size_field = "total_byte_size"
        else:
            size_metric = None
            size_field = None
        parts = _partitioning_plan(fragments, size_metric, size_field, granularity).to_records(index=False)

        return parts, (None,) * (len(parts) + 1)
    

    @classmethod
    def make_partition(cls, part, **kwargs):
        raise NotImplementedError
    
    
class ArrowEngine(Engine):

    @classmethod
    def _make_meta(cls, dataset):
        return dataset.schema.empty_table().to_pandas()
    
    @classmethod
    def gather_metadata(
        cls,
        urlpath,
        columns=None,
        index=None,
        filters=None,
        engine=None,
        blocksize="auto",
        chunksize=None,
        granularity="row-group",
        groups=None,
        storage_options=None,
        dataset_options=None,
        read_options=None,
    ):
        
        if chunksize:
            blocksize = None
        elif blocksize == "auto":
            blocksize = _BLOCKSIZE_DEFAULT
        elif blocksize:
            blocksize = parse_bytes(blocksize)
        
        storage_options = storage_options or {}
        dataset_options = dataset_options or {}
        read_options = read_options or {}
        fs, _, paths = get_fs_token_paths(urlpath, mode="rb", storage_options=storage_options)
        if not isinstance(urlpath, (list, tuple)):
            paths = sorted(paths, key=natural_sort_key)
        if not paths:
            raise ValueError
        
        # Process filters argument
        ds_filters = None
        if isinstance(filters, pa.compute.Expression):
            ds_filters = filters
        elif filters is not None:
            ds_filters = pq._filters_to_expression(filters)
        
        
        # Create pyarrow dataset
        if "filesystem" not in dataset_options:
            dataset_options["filesystem"] = fs
        if "format" not in dataset_options:
            dataset_options["format"] = pa_ds.ParquetFileFormat()
        dataset = None
        if len(paths) == 1 and fs.isdir(paths[0]):
            meta_path = fs.sep.join([paths[0], "_metadata"])
            if fs.exists(meta_path):
                # Use _metadata file
                dataset = pa_ds.parquet_dataset(meta_path, **dataset_options)
        if dataset is None:
            dataset = pa_ds.dataset(paths, **dataset_options)


        # Gather file metadata
        # TODO: Avoid collecting unnecessary info when
        # granularity="file" and/or blocksize=chunksize=None
        def _make_record(rg_frag, path, chunksize, blocksize):
            row_group = rg_frag.row_groups[0]
            record = {
                "path": path,
                "row_group": row_group.id,
            }
            if chunksize:
                record["num_rows"] = row_group.num_rows
            elif blocksize:
                record["total_byte_size"] = row_group.total_byte_size
            return record

        def _make_file_records(file_frag, chunksize, blocksize, granularity, ds_filters):

            if granularity == "file":
                # Avoid splitting by row-group
                if chunksize:
                    return[{"path": file_frag.path, "row_group": None, "num_rows": file_frag.count_rows()}]
                elif blocksize:
                    return[{"path": file_frag.path, "row_group": None, "total_byte_size": sum(rg.total_byte_size for rg in file_frag.row_groups)}]
                else:
                    # Simple case
                    return[{"path": file_frag.path, "row_group": None}]
            else:
                return [
                    _make_record(rg_frag, file_frag.path, chunksize, blocksize)
                    for rg_frag in file_frag.split_by_row_group(ds_filters)
                ]

        if _is_local_fs(fs):
            fragment_info = list(
                itertools.chain(
                    *[
                        _make_file_records(file_frag, chunksize, blocksize, granularity, ds_filters)
                        for file_frag in dataset.get_fragments(ds_filters)
                    ]
                )
            )
        else:
            _make_file_records_delayed = delayed(_make_file_records)
            fragment_info = list(
                itertools.chain(
                    *dask.compute(
                        [
                            _make_file_records_delayed(file_frag, chunksize, blocksize, granularity, ds_filters)
                            for file_frag in dataset.get_fragments(ds_filters)
                        ]
                    )[0]
                )
            )
        fragments = pd.DataFrame.from_records(fragment_info)


        # Define output `meta`
        # TODO: Deal with index and partitioned datasets
        meta = cls._make_meta(dataset)
        if columns:
            columns = meta.columns
        if meta.index.names != [None]:
            meta = meta.reset_index()
        if index:
            meta = meta.set_index(index)
        index = meta.index.names
        
        partitioning_options = {
            "index": index,
            "chunksize": chunksize,
            "blocksize": blocksize,
            "granularity": granularity,
        }
        map_options = {
            "fs": fs, 
            "columns": columns,
            "index": index,
        }
        
        return (
            meta,
            fragments,
            partitioning_options,
            read_options,
            map_options,
        )


    @classmethod
    def make_partition(cls, part, **kwargs):
        raise NotImplementedError


class CudfEngine(ArrowEngine):

    @classmethod
    def _make_meta(cls, dataset):
        return cudf.DataFrame.from_arrow(dataset.schema.empty_table())

    
    @classmethod
    def make_partition(cls, part, **kwargs):
        reads = pd.DataFrame.from_dict({"path": part[0], "rgs": part[1]}).groupby("path").agg(list)
        paths = reads.index.tolist()
        row_groups = None if (part[1] and part[1][0] is None) else reads.rgs.tolist()
        open_file_options = None
        fs = kwargs.pop("fs")
        columns = kwargs.pop("columns", None)
        index = kwargs.pop("index", None)

        with ExitStack() as stack:

            # Non-local filesystem handling
            paths_or_fobs = paths
            if not _is_local_fs(fs):
                paths_or_fobs = _open_remote_files(
                    paths_or_fobs,
                    fs,
                    context_stack=stack,
                    **_default_open_file_options(
                        open_file_options, columns, row_groups
                    ),
                )

            # Use cudf to read in data
            df = cudf.read_parquet(
                paths_or_fobs,
                columns=columns,
                row_groups=row_groups,
                **kwargs,
            )

        # TODO: Index logic
        return df


def read_parquet(
    urlpath,
    columns=None,  # TODO
    index=None,  # TODO
    filters=None,
    #
    engine="pyarrow",
    #
    blocksize="auto",  # max partition storage size
    chunksize=None,  # partition row-count (takes precedence over `blocksize`)
    #
    granularity="row-group",  # Options: {"row-group", "file"}
    groups=None,  # TODO
    #
    storage_options=None,
    dataset_options=None,
    **read_options,
):
  
    ### SET ENGINE
    
    engine = CudfEngine if engine == "cudf" else ArrowEngine  # TODO: Proper engine dispatching

    ### PROCESS METADATA (Overriding is Required)
    (
        meta,
        fragments,
        partitioning_options,
        read_options,
        map_options,
    ) = engine.gather_metadata(
        urlpath,
        columns=columns,
        index=index,
        filters=filters,
        engine=engine,
        blocksize=blocksize,
        chunksize=chunksize,
        granularity=granularity,
        groups=groups,
        dataset_options=dataset_options,
        read_options=read_options,
        storage_options=storage_options,
    )

    
    ### GENERATE PARTITIONING PLAN (Overriding is Optional)
    parts, divisions = engine.make_partitioning_plan(fragments, partitioning_options)
    
    
    ### CALL FROM_MAP (ENGINE AGNOSTIC)
    return dd.from_map(
        ReadParquet(engine, meta, map_options),
        parts,
        meta=meta,
        divisions=divisions,
        label="read-parquet",
        enforce_metadata=False,
        **read_options,
    )