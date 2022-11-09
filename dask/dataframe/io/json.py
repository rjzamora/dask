import io
import os

import pandas as pd
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block

from dask.base import compute as dask_compute
from dask.bytes.core import byte_ranges
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_map
from dask.dataframe.utils import insert_meta_param_description, make_meta
from dask.delayed import delayed


def to_json(
    df,
    url_path,
    orient="records",
    lines=None,
    storage_options=None,
    compute=True,
    encoding="utf-8",
    errors="strict",
    compression=None,
    compute_kwargs=None,
    name_function=None,
    **kwargs,
):
    """Write dataframe into JSON text files

    This utilises ``pandas.DataFrame.to_json()``, and most parameters are
    passed through - see its docstring.

    Differences: orient is 'records' by default, with lines=True; this
    produces the kind of JSON output that is most common in big-data
    applications, and which can be chunked when reading (see ``read_json()``).

    Parameters
    ----------
    df: dask.DataFrame
        Data to save
    url_path: str, list of str
        Location to write to. If a string, and there are more than one
        partitions in df, should include a glob character to expand into a
        set of file names, or provide a ``name_function=`` parameter.
        Supports protocol specifications such as ``"s3://"``.
    encoding, errors:
        The text encoding to implement, e.g., "utf-8" and how to respond
        to errors in the conversion (see ``str.encode()``).
    orient, lines, kwargs
        passed to pandas; if not specified, lines=True when orient='records',
        False otherwise.
    storage_options: dict
        Passed to backend file-system implementation
    compute: bool
        If true, immediately executes. If False, returns a set of delayed
        objects, which can be computed at a later time.
    compute_kwargs : dict, optional
        Options to be passed in to the compute method
    encoding, errors:
        Text conversion, ``see str.encode()``
    compression : string or None
        String like 'gzip' or 'xz'.
    name_function : callable, default None
        Function accepting an integer (partition index) and producing a
        string to replace the asterisk in the given filename globstring.
        Should preserve the lexicographic order of partitions.
    """
    if lines is None:
        lines = orient == "records"
    if orient != "records" and lines:
        raise ValueError(
            "Line-delimited JSON is only available with" 'orient="records".'
        )
    kwargs["orient"] = orient
    kwargs["lines"] = lines and orient == "records"
    outfiles = open_files(
        url_path,
        "wt",
        encoding=encoding,
        errors=errors,
        name_function=name_function,
        num=df.npartitions,
        compression=compression,
        **(storage_options or {}),
    )
    parts = [
        delayed(write_json_partition)(d, outfile, kwargs)
        for outfile, d in zip(outfiles, df.to_delayed())
    ]
    if compute:
        if compute_kwargs is None:
            compute_kwargs = dict()
        return list(dask_compute(*parts, **compute_kwargs))
    else:
        return parts


def write_json_partition(df, openfile, kwargs):
    with openfile as f:
        df.to_json(f, **kwargs)
    return os.path.normpath(openfile.path)


class JsonReader:
    """
    JsonReader Class

    Reads JSON data from disk to produce a partition.
    """

    def __init__(
        self,
        fs,
        paths=None,
        sample=2**20,
        path_converter=None,
        encoding="utf-8",
        errors="strict",
        engine=pd.read_json,
        include_path_column="path",
        meta=None,
        delimiter=b"\n",
        compression="infer",
        **kwargs,
    ):
        self.fs = fs
        self.encoding = encoding
        self.errors = errors
        self.engine = engine
        self.include_path_column = include_path_column
        self.delimiter = delimiter
        self.compression = compression
        self.path_converter = path_converter
        self.kwargs = kwargs

        # Use list of paths to define categorical path_dtype
        if include_path_column:
            if paths is None:
                raise ValueError("paths required to support include_path_column")
            if path_converter:
                self.path_dtype = pd.CategoricalDtype(path_converter(p) for p in paths)
            else:
                self.path_dtype = pd.CategoricalDtype(paths)
        else:
            self.path_dtype = None

        # Sample or read data to define `meta`
        self.meta = meta  # Temporary assignment required
        if meta is None:
            if sample:
                meta = self.read_chunk(paths[0], 0, sample)
            else:
                meta = self.read_file(paths[0])
        self.meta = make_meta(meta)

    def __call__(self, part):

        # Expand part into a tuple
        path, offset, length = part

        # Read chunk or entire file
        if length is not None:
            return self.read_chunk(path, offset, length)
        else:
            return self.read_file(path)

    def add_path_column(self, df, path):
        column_name = self.include_path_column
        path_dtype = self.path_dtype
        path_name = self.path_converter(path) if self.path_converter else path
        if column_name:
            if column_name in df.columns:
                raise ValueError(
                    f"Files already contain the column name: '{column_name}', so the path "
                    "column cannot use this name. Please set `include_path_column` to a "
                    "unique name."
                )
            df = df.assign(
                **{
                    column_name: df._constructor_sliced(
                        [path_name] * len(df),
                        dtype=path_dtype,
                    )
                }
            )
        return df

    def read_chunk(self, path, offset, length):
        with OpenFile(self.fs, path, mode="rb", compression=self.compression) as f:
            chunk = read_block(
                f,
                offset,
                length,
                self.delimiter,
            )
        s = io.StringIO(chunk.decode(self.encoding, self.errors))
        s.seek(0)
        df = self.engine(s, **self.kwargs)
        if df.empty and self.meta is not None:
            return self.meta
        return self.add_path_column(df, path)

    def read_file(self, path):
        with OpenFile(self.fs, path, mode="rb", compression=self.compression) as f:
            df = self.engine(f, **self.kwargs)
        return self.add_path_column(df, path)


@dataframe_creation_dispatch.register_inplace("pandas")
@insert_meta_param_description
def read_json(
    url_path,
    orient="records",
    lines=None,
    storage_options=None,
    blocksize=None,
    sample=2**20,
    encoding="utf-8",
    errors="strict",
    compression="infer",
    meta=None,
    engine=pd.read_json,
    engine_wrapper=None,
    include_path_column=False,
    path_converter=None,
    **kwargs,
):
    """Create a dataframe from a set of JSON files

    This utilises ``pandas.read_json()``, and most parameters are
    passed through - see its docstring.

    Differences: orient is 'records' by default, with lines=True; this
    is appropriate for line-delimited "JSON-lines" data, the kind of JSON output
    that is most common in big-data scenarios, and which can be chunked when
    reading (see ``read_json()``). All other options require blocksize=None,
    i.e., one partition per input file.

    Parameters
    ----------
    url_path: str, list of str
        Location to read from. If a string, can include a glob character to
        find a set of file names.
        Supports protocol specifications such as ``"s3://"``.
    encoding, errors:
        The text encoding to implement, e.g., "utf-8" and how to respond
        to errors in the conversion (see ``str.encode()``).
    orient, lines, kwargs
        passed to pandas; if not specified, lines=True when orient='records',
        False otherwise.
    storage_options: dict
        Passed to backend file-system implementation
    blocksize: None or int
        If None, files are not blocked, and you get one partition per input
        file. If int, which can only be used for line-delimited JSON files,
        each partition will be approximately this size in bytes, to the nearest
        newline character.
    sample: int
        Number of bytes to pre-load, to provide an empty dataframe structure
        to any blocks without data. Only relevant when using blocksize.
    encoding, errors:
        Text conversion, ``see bytes.decode()``
    compression : string or None
        String like 'gzip' or 'xz'.
    engine : function object, default ``pd.read_json``
        The underlying function that dask will use to read JSON files. By
        default, this will be the pandas JSON reader (``pd.read_json``).
    engine_wrapper : Callable, default ``JsonReader``
        Alternative wrapper-class to use instead of ``JsonReader``.
    include_path_column : bool or str, optional
        Include a column with the file path where each row in the dataframe
        originated. If ``True``, a new column is added to the dataframe called
        ``path``. If ``str``, sets new column name. Default is ``False``.
    path_converter : function or None, optional
        A function that takes one argument and returns a string. Used to convert
        paths in the ``path`` column, for instance, to strip a common prefix from
        all the paths.
    $META

    Returns
    -------
    dask.DataFrame

    Examples
    --------
    Load single file

    >>> dd.read_json('myfile.1.json')  # doctest: +SKIP

    Load multiple files

    >>> dd.read_json('myfile.*.json')  # doctest: +SKIP

    >>> dd.read_json(['myfile.1.json', 'myfile.2.json'])  # doctest: +SKIP

    Load large line-delimited JSON files using partitions of approx
    256MB size

    >> dd.read_json('data/file*.csv', blocksize=2**28)
    """
    if lines is None:
        lines = orient == "records"
    if orient != "records" and lines:
        raise ValueError(
            "Line-delimited JSON is only available with" 'orient="records".'
        )
    if blocksize and (orient != "records" or not lines):
        raise ValueError(
            "JSON file chunking only allowed for JSON-lines"
            "input (orient='records', lines=True)."
        )
    storage_options = storage_options or {}
    if include_path_column is True:
        include_path_column = "path"

    if not isinstance(url_path, (str, list, tuple, os.PathLike)):
        raise TypeError("Path should be a string, os.PathLike, list or tuple")

    fs, fs_token, paths = get_fs_token_paths(
        url_path, mode="rb", storage_options=kwargs
    )

    if len(paths) == 0:
        raise OSError("%s resolved to no files" % url_path)

    if blocksize:
        offsets, lengths = byte_ranges(
            paths,
            fs,
            blocksize=blocksize,
            compression=compression,
        )

        parts = [
            (path, o, l)
            for path, path_offsets, path_lengths in zip(paths, offsets, lengths)
            for o, l in zip(path_offsets, path_lengths)
        ]
    else:
        parts = [(path, 0, None) for path in paths]

    reader = (engine_wrapper or JsonReader)(
        fs,
        paths=paths,
        sample=sample if blocksize else 0,
        path_converter=path_converter,
        encoding=encoding,
        errors=errors,
        engine=engine,
        include_path_column=include_path_column,
        meta=meta,
        delimiter=b"\n",
        compression=compression,
        lines=lines,
        orient=orient,
        **kwargs,
    )

    return from_map(
        reader,
        parts,
        meta=reader.meta,
        label="read-json",
        enforce_metadata=False,
    )
