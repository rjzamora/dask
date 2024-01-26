import contextlib
import glob
import os
import pickle
from collections import defaultdict
from importlib import import_module

import dask.dataframe as dd
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.utils import typename

from distributed import get_worker, wait, default_client
from distributed.protocol import dask_deserialize, dask_serialize


class Handler:
    """Base class for format-specific checkpointing handlers
    
    A ``Handler`` object will be responsible for a single partition.
    """
    format = None  # General format label

    def __init__(self, path, backend, index, **kwargs):
        self.path = path
        self.backend = backend
        self.index = index
        self.kwargs = kwargs

    @classmethod
    def clean(cls, dirpath):
        """Clean the target directory"""
        import shutil
    
        if os.path.isdir(dirpath):
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(dirpath)
    
    @classmethod
    def prepare(cls, dirpath):
        """Create the target directory"""
        os.makedirs(dirpath, exist_ok=True)

    @classmethod
    def save(cls, part, path, index, id):
        """Persist the target partition to disk"""
        raise NotImplementedError()  # Logic depends on format

    def load(self):
        """Collect the saved partition"""
        raise NotImplementedError()  # Logic depends on format


@dask_serialize.register(Handler)
def _serialize_unloaded(obj):
    # Make sure we read the partition into memory if
    # this partition is moved to a different worker
    return None, [pickle.dumps(obj.load())]


@dask_deserialize.register(Handler)
def _deserialize_unloaded(header, frames):
    # Deserializing a `Handler` object returns the wrapped data
    return pickle.loads(frames[0])


class ParquetHandler(Handler):
    """Parquet-specific checkpointing handler for DataFrame collections"""
    format = "parquet"
    
    @classmethod
    def save(cls, part, path, index):
        fn = f"{path}/part.{index[0]}.parquet"
        part.to_parquet(fn)
        # return get_worker().worker_address
        return index[0]

    def load(self):
        lib = import_module(self.backend)
        fn = glob.glob(f"{self.path}/*.{self.index}.parquet")
        return lib.read_parquet(fn, **self.kwargs)


class Checkpoint:
    """Checkpoint a Dask collection on disk
    
    The storage location does not need to be shared between workers.
    """

    def __init__(
        self,
        npartitions,
        meta,
        handler,
        path,
        id,
        load_kwargs,
    ):
        self.npartitions = npartitions
        self.meta = meta
        self.backend = typename(meta).partition(".")[0]
        self.handler = handler
        self.path = path
        self.id = id
        self.load_kwargs = load_kwargs or {}
        self._valid = True
    
    def __repr__(self):
        ctype = type(self.meta).__name__
        path = self.path
        fmt = self.handler.format
        return f"Checkpoint({ctype})<path={path}, format={fmt}>"

    def __del__(self):
        self.clean()

    @classmethod
    def create(
        cls,
        df,
        dirpath,
        id=None,
        format="parquet",
        overwrite=False,
        compute_kwargs=None,
        load_kwargs=None,
        **save_kwargs,
    ):
        """Create a new Checkpoint object"""

        # Get handler
        if format == "parquet":
            handler = ParquetHandler
        else:
            # Only parquet supported for now
            raise NotImplementedError()

        id = id or tokenize(df, dirpath, format)
        meta = df._meta

        client = default_client()
        
        wait(client.run(handler.prepare, dirpath))
        path = f"{dirpath}/{id}"

        if overwrite:
            wait(client.run(handler.clean, path))

        wait(client.run(handler.prepare, path))

        npartitions = len(
            df.map_partitions(
                handler.save,
                path,
                BlockIndex((df.npartitions,)),
                meta=meta,
                enforce_metadata=False,
                **save_kwargs,
            ).compute(**(compute_kwargs or {}))
        )

        return cls(
            npartitions,
            meta,
            handler,
            path,
            id,
            load_kwargs,
        )

    def load(self):
        """Load a checkpointed collection
        
        Note that this will not immediately persist the partitions
        in memory. Rather, it will output a lazy Dask collection.
        """
        if not self._valid:
            raise RuntimeError("This checkpoint is no longer valid")

        #
        # Get client and check workers
        #
        client = default_client()

        #
        # Find out which partition indices are stored on each worker
        #
        def get_indices(path):
            # Assume file-name is something like: <name>.<index>.<fmt>
            return {int(fn.split(".")[-2]) for fn in glob.glob(path + "/*")}

        worker_indices = client.run(get_indices, self.path)
    
        summary = defaultdict(list)
        for worker, indices in worker_indices.items():
            for index in indices:
                summary[index].append(worker)

        assert len(summary) == self.npartitions, "Load failed."

        #
        # Convert each checkpointed partition to a `Handler` object
        #
        assignments = {}
        futures = []
        for i, (worker, indices) in enumerate(summary.items()):
            assignments[worker] = indices[i % len(indices)]
            futures.append(
                client.submit(
                    self.handler,
                    self.path,
                    self.backend,
                    i,
                    workers=[assignments[i]],
                    **self.load_kwargs,
                )
            )
        wait(futures)

        #
        # Crate a new collection from the delayed `Handler` objects
        #
        meta = self.meta
        return dd.from_delayed(futures, meta=meta, verify_meta=False).map_partitions(
            self._load_partition,
            meta=meta,
        )

    @staticmethod
    def _load_partition(obj):
        """Load a checkpointed partition"""
        if isinstance(obj, Handler):
            return obj.load()
        return obj

    def clean(self):
        """Clean up this checkpoint"""
        client = default_client()
        wait(client.run(self.handler.clean, self.path))
        self._valid = False


checkpoint = Checkpoint.create  # Expose "public" API