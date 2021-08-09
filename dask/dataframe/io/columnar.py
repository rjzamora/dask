import warnings


class BaseEngine:
    """The API necessary to provide a new 'columnar' reader/writer

    This API is currently used for both Parquet and ORC datasets.
    """

    @classmethod
    def read_metadata(cls, *args, **kwargs):
        """Gather metadata for a columnar dataset to prepare for a read

        This method is deprecated in favor of ``plan_read``, and will
        removed in a future Dask release.
        """
        raise NotImplementedError()

    @classmethod
    def plan_read(cls, fs, paths, **kwargs):
        """Gather metadata for a columnar dataset to prepare for a read

        This function is called once in the user's Python session to gather
        important metadata about the dataset.

        Parameters
        ----------
        fs: FileSystem
        paths: List[str]
            A list of paths to files (or their equivalents)
        columns: list or None
            Columns to include in the read.  If ``None``, we assume all
            available columns are desired.
        categories: list, dict or None
            Column(s) containing categorical data.
        index: str, List[str], or False
            The column name(s) to be used as the index.
            If set to ``None``, pandas metadata (if available) can be used
            to reset the value in this function
        gather_statistics: bool
            Whether or not to gather statistics data.  If ``None``, we only
            gather statistics data if there is a global _metadata file
            available to query (cheaply)
        filters: list or None
            List of filters to apply, like ``[('x', '>', 0), ...]``.
        split_row_groups: bool or int
            Whether to create a separate output partition for every
            contiguously-stored row-group/stripe in the file. If an integer
            is specified, each output partition will correspond to that
            number of row-groups/stripes (or fewer).
        **kwargs: dict (of dicts)
            User-specified arguments to pass on to backend.
            Top level key can be used by engine to select appropriate dict.

        Returns
        -------
        meta: pandas.DataFrame
            An empty DataFrame object to use for metadata.
            Should have appropriate column names and dtypes but need not have
            any actual data
        statistics: Optional[List[Dict]]
            Either None, if no statistics were found, or a list of dictionaries
            of statistics data, one dict for every partition (see the next
            return value).  The statistics should look like the following:

            [
                {'num-rows': 1000, 'columns': [
                    {'name': 'id', 'min': 0, 'max': 100},
                    {'name': 'x', 'min': 0.0, 'max': 1.0},
                    ]},
                ...
            ]
        parts: List[object]
            A list of objects to be passed to ``Engine.read_partition``.
            Each object should represent a piece of data (usually a row-group).
            The type of each object can be anything, as long as the
            engine's read_partition function knows how to interpret it.
        """

        # Compatibility Logic:
        #
        # If we get here at run time, the engine has NOT defined
        # a `plan_read` method yet. Therefore, we should fall back
        # to the ("legacy") `read_metadata` definition and convert
        # the output to a dictionary

        warnings.warn(
            "Using an Engine-derived class where `plan_read` is not "
            "defined. We will fall back on `read_metadata` for now. "
            "Please define `plan_read`, since `read_metadata` will be "
            "removed in a future version of Dask.",
            FutureWarning,
        )

        # Use "legacy" read_metadata class method
        meta, statistics, parts, index = cls.read_metadata(fs, paths, **kwargs)[:4]
        common_kwargs = {}
        aggregation_depth = False
        if len(parts):
            # `common_kwargs` and `aggregation_depth`
            # may be stored in the first element of `parts`
            common_kwargs = parts[0].pop("common_kwargs", {})
            aggregation_depth = parts[0].pop("aggregation_depth", aggregation_depth)

        return {
            "meta": meta,
            "statistics": statistics,
            "parts": parts,
            "index": index,
            "common_kwargs": common_kwargs,
            "aggregation_depth": aggregation_depth,
        }

    @classmethod
    def read_partition(cls, fs, piece, columns, index, **kwargs):
        """Read a single piece of a columnar dataset into a Pandas DataFrame

        This function may be called multiple times within a single
        task if

        Parameters
        ----------
        fs: FileSystem
        piece: object
            This is some token that is returned by Engine.read_metadata.
            Typically it represents a row group in a Parquet dataset
        columns: List[str]
            List of column names to pull out of that row group
        index: str, List[str], or False
            The index name(s).
        **kwargs:
            Includes `"kwargs"` values stored within the `parts` output
            of `engine.read_metadata`. May also include arguments to be
            passed to the backend (if stored under a top-level `"read"` key).

        Returns
        -------
        A Pandas DataFrame
        """
        raise NotImplementedError()

    @classmethod
    def initialize_write(
        cls,
        df,
        fs,
        path,
        append=False,
        partition_on=None,
        ignore_divisions=False,
        division_info=None,
        **kwargs,
    ):
        """Perform engine-specific initialization steps for this dataset

        Parameters
        ----------
        df: dask.dataframe.DataFrame
        fs: FileSystem
        path: str
            Destination directory for data.  Prepend with protocol like ``s3://``
            or ``hdfs://`` for remote data.
        append: bool
            If True, may use existing metadata (if any) and perform checks
            against the new data being stored.
        partition_on: List(str)
            Column(s) to use for dataset partitioning in parquet.
        ignore_divisions: bool
            Whether or not to ignore old divisions when appending.  Otherwise,
            overlapping divisions will lead to an error being raised.
        division_info: dict
            Dictionary containing the divisions and corresponding column name.
        **kwargs: dict
            Other keyword arguments (including `index_cols`)

        Returns
        -------
        tuple:
            engine-specific instance
            list of filenames, one per partition
        """
        raise NotImplementedError

    @classmethod
    def write_partition(
        cls, df, path, fs, filename, partition_on, return_metadata, **kwargs
    ):
        """
        Output a partition of a dask.DataFrame. This will correspond to
        one output file, unless partition_on is set, in which case, it will
        correspond to up to one file in each sub-directory.

        Parameters
        ----------
        df: dask.dataframe.DataFrame
        path: str
            Destination directory for data.  Prepend with protocol like ``s3://``
            or ``hdfs://`` for remote data.
        fs: FileSystem
        filename: str
        partition_on: List(str)
            Column(s) to use for dataset partitioning in parquet.
        return_metadata : bool
            Whether to return list of instances from this write, one for each
            output file. These will be passed to write_metadata if an output
            metadata file is requested.
        **kwargs: dict
            Other keyword arguments (including `fmd` and `index_cols`)

        Returns
        -------
        List of metadata-containing instances (if `return_metadata` is `True`)
        or empty list
        """
        raise NotImplementedError

    @classmethod
    def write_metadata(cls, parts, meta, fs, path, append=False, **kwargs):
        """
        Write the shared metadata file for a parquet dataset.

        Parameters
        ----------
        parts: List
            Contains metadata objects to write, of the type undrestood by the
            specific implementation
        meta: non-chunk metadata
            Details that do not depend on the specifics of each chunk write,
            typically the schema and pandas metadata, in a format the writer
            can use.
        fs: FileSystem
        path: str
            Output file to write to, usually ``"_metadata"`` in the root of
            the output dataset
        append: boolean
            Whether or not to consolidate new metadata with existing (True)
            or start from scratch (False)
        **kwargs: dict
            Other keyword arguments (including `compression`)
        """
        raise NotImplementedError()

    @classmethod
    def collect_file_metadata(cls, path, fs, file_path):
        """
        Collect parquet metadata from a file and set the file_path.

        Parameters
        ----------
        path: str
            Parquet-file path to extract metadata from.
        fs: FileSystem
        file_path: str
            Relative path to set as `file_path` in the metadata.

        Returns
        -------
        A metadata object.  The specific type should be recognized
        by the aggregate_metadata method.
        """
        raise NotImplementedError()

    @classmethod
    def aggregate_metadata(cls, meta_list, fs, out_path):
        """
        Aggregate a list of metadata objects and optionally
        write out the final result as a _metadata file.

        Parameters
        ----------
        meta_list: list
            List of metadata objects to be aggregated into a single
            metadata object, and optionally written to disk. The
            specific element type can be engine specific.
        fs: FileSystem
        out_path: str or None
            Directory to write the final _metadata file. If None
            is specified, the aggregated metadata will be returned,
            and nothing will be written to disk.

        Returns
        -------
        If out_path is None, an aggregate metadata object is returned.
        Otherwise, None is returned.
        """
        raise NotImplementedError()
