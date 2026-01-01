from dataclasses import dataclass
import time
from pathlib import Path
import re
from typing import Mapping, Set, Any, Literal, Dict, Optional, Tuple
from jinja2 import Environment, BaseLoader, StrictUndefined, meta
import pandas as pd
from google.cloud import bigquery
from .forecast.checks import _is_pandas_df, _is_polars_df


def read_sql(sql_path: Path) -> str:
    """Reads in a .sql file at the given sql_path into a string and returns the query string.

    Args:
        sql_path (Path): Path object of .sql file to read.

    Raises:
        FileNotFoundError: SQL file not found at the given sql_path

    Returns:
        str: Contents of the .sql file at sql_path as a string
    """
    try:
        with open(
            sql_path, "r", encoding="utf-8", errors="strict", newline=None
        ) as file_handle:
            return file_handle.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"SQL file not found: {sql_path}")


def replace_params_sql(sql_text: str, replace_dict: dict[str, str]) -> str:
    """In a string (SQL query) replace a placeholder denoted by <<placeholder_name>> with
    a corresponding string and return the resulting string (SQL query) after all replacements
    have been made.

    Args:
        sql_text (str): SQL query represented as a string. Initially has placeholders
            <<placeholder_name>>
        replace_dict (dict[str,str]): Mapping dictionary giving map of
            "placeholder_name":"placeholder_value"
            where "placeholder_name" is the value to be replaced by the new string
            "placeholder_value".

    Raises:
        KeyError: Raise KeyError if no instances of a given placeholder are found in `sql_text`
        ValueError: Raise ValueError if there are remaining placeholders that have not been
            replaced in `sql_text`.

    Returns:
        str: SQL query represented as a string with all <<placeholder_name>> strings replaced.
    """
    for placeholder_name, replace_placeholder_with_text in replace_dict.items():
        # define expression pattern
        pattern = re.compile(rf"<<\s*{re.escape(placeholder_name)}\s*>>")

        # Check if there are no instances of placeholder_name & raise error if no instances
        if not pattern.search(sql_text):
            raise KeyError(
                f"Placeholder '<<{placeholder_name}>>' not found in SQL query"
            )

        # for a given pattern, substitute
        sql_text = pattern.sub(lambda _: replace_placeholder_with_text, sql_text)

    # Check if left over placeholders <<>> and if so, raise ValueError
    remaining_placeholders = {
        match.group(1) for match in re.finditer(r"<<\s*([A-Za-z_]\w*)\s*>>", sql_text)
    }
    if remaining_placeholders:
        raise ValueError(f"Placeholders still remain: {remaining_placeholders}")
    return sql_text


def _vars_in_template(env: Environment, sql_text: str) -> Set[str]:
    """Extract undeclared variable names used in a Jinja template.

    Parses the given SQL/Jinja text with the provided Jinja `Environment` and
    returns the set of *top-level* variable names that must be supplied in the
    render context. Common Jinja builtins/sentinels (e.g., ``loop``) are
    filtered out.

    Args:
      env: A Jinja2 `Environment` used to parse the template source.
      sql_text(str): Raw SQL text containing Jinja placeholders and control blocks.

    Returns:
      A set of variable names referenced by the template that are not defined
      within the template itself (i.e., must be provided at render time).

    Notes:
      The returned set excludes typical Jinja builtins/sentinels such as
      ``True``, ``False``, ``None``, ``loop``, ``cycler``, and ``namespace``.

    Examples:
      >>> env = Environment(loader=BaseLoader())
      >>> _vars_in_template(env, "SELECT * FROM {{ schema }}.{{ table }}")
      {'schema', 'table'}
    """
    jinja_ast = env.parse(sql_text)
    vars_used = meta.find_undeclared_variables(jinja_ast)

    # Filter out common Jinja builtins/sentinels that may appear in meta scan
    builtins = {"True", "False", "None", "loop", "cycler", "namespace"}
    return {var for var in vars_used if var not in builtins}


def render_sql_template(sql_text: str, params: Mapping[str, object]) -> str:
    """
    Render a Jinja SQL template, enforcing:
      1) Every placeholder used in the SQL must be present in `params`.
      2) Every key in `params` must be used in the SQL.

    Notes:
      - This check is independent of any `| default(...)` filters in the SQL:
        even if a variable has a Jinja default, we still require it in `params`.
      - With StrictUndefined, nested/attribute lookups like {{ obj.attr }}
        still raise if structure doesn't match what the template expects.

    Args:
      sql_text (str): Raw SQL text containing Jinja placeholders and/or control
        blocks (e.g., ``{{ var }}``, ``{% if ... %}``).
      params (dict(str,str)): Dictionary mapping of
       placeholder_name: placeholder_value. Used to render the template.

    Returns:
      The fully rendered SQL string.

    Raises:
      ValueError: If there is a parameter mismatch:
        - **missing**: placeholders used in the template but not present in
          ``params``.
        - **unused**: keys present in ``params`` but not referenced in the
          template.
      jinja2.exceptions.UndefinedError: If, during rendering with
        ``StrictUndefined``, the template accesses an undefined variable or a
        missing attribute/key on a provided object.

    Examples:
      Basic usage:

      >>> sql = "SELECT * FROM `proj.{{ schema }}.{{ table }}`"
      >>> render_sql_template(sql, {"schema": "sales", "table": "events"})
      'SELECT * FROM `proj.sales.events`'
    """
    # Build a non-loading env for static analysis + a rendering env.
    analyze_env = Environment(loader=BaseLoader())
    needed = _vars_in_template(analyze_env, sql_text)

    supplied = set(params.keys())
    missing = needed - supplied
    extras = supplied - needed

    if missing or extras:
        problems = []
        if missing:
            problems.append(f"missing: {sorted(missing)}")
        if extras:
            problems.append(f"unused: {sorted(extras)}")

        msg = f"Jinja render error: Parameter mismatch for SQL template ({'; '.join(problems)})."
        raise ValueError(msg)

    # Render with strict undefined to catch structural mistakes at runtime.
    render_env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # raises error if key in params is undefined
    )
    template = render_env.from_string(sql_text)
    return template.render(**params)


@dataclass(frozen=True)
class BiqQueryQueryStats:
    job_id: str
    total_rows: Optional[int]
    total_bytes_processed: Optional[int]
    total_bytes_billed: Optional[int]
    cache_hit: Optional[bool]
    elapsed_seconds: float


def query_to_dataframe(
    sql: str,
    *,
    client: bigquery.Client,
    job_config: bigquery.QueryJobConfig,
    dataframe_type: Literal["pandas", "polars"] = "pandas",
    use_bqstorage: bool = True,
) -> Tuple[object, BiqQueryQueryStats]:
    start = time.perf_counter()

    job = client.query(sql, job_config=job_config)
    result = job.result()

    elapsed = time.perf_counter() - start

    if dataframe_type == "pandas":
        import pandas as pd

        df = result.to_dataframe(create_bqstorage_client=use_bqstorage)

    elif dataframe_type == "polars":
        pass


def _check_storage_uri_str(storage_uri_str: str, uri_prefix: str = "gs://") -> None:
    """Validate cloud storage URI

    Args:
      storage_uri_str (str): The storage URI to check.
      uri_prefix (str): Expected URI prefix (default: "gs://").

    Returns:
      None

    Raises:
      ValueError: If given URI is not a string or does not contain the URI prefix

    """
    if not isinstance(storage_uri_str, str) or not storage_uri_str.startswith(
        uri_prefix
    ):
        raise ValueError(
            f"storage_uri must be a {uri_prefix}... string, got {storage_uri_str!r}."
        )


def _check_gcs_file_stats(
    gs_uri: str,
    filesystem: str = "gcs",
    uri_prefix: str = "gs://",
    storage_options: Mapping[str, Any] | None = None,
    filesystem_obj: Any | None = None,
) -> Mapping[str, Any]:
    """Return object metadata for a cloud/path via fsspec `fs.info(...)`.

    Args:
      gs_uri: Absolute object URI (e.g., "gs://my-bucket/path/to/file.parquet").
      filesystem: fsspec filesystem name (default: "gcs").
      storage_options: Options forwarded to `fsspec.filesystem(...)`
        (e.g., {"token": "cloud"}).
      filesystem_obj: Pre-created filesystem instance, if you already have one
        (e.g., tests: `fsspec.filesystem("memory")`).

    Returns:
      A filesystem-specific info dictionary describing the object.

    Raises:
      ValueError: If `filesystem == "gcs"` and `gs_uri` does not start with "gs://".
      FileNotFoundError: If the object doesn't exist.
      RuntimeError: If required packages aren't installed.
    """

    _check_storage_uri_str(gs_uri, uri_prefix)

    try:
        if filesystem_obj is None:
            import fsspec

            storage_options = dict(storage_options or {})

            filesystem_obj = fsspec.filesystem(filesystem, **storage_options)
    except ImportError as e:
        raise RuntimeError(
            f"Checking {filesystem} stats requires package fsspec which could not be imported"
        ) from e

    try:
        info = filesystem_obj.info(gs_uri)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Object not found: {gs_uri} fs={filesystem}") from e
    return info


def write_df_to_gcs_parquet(
    df,
    gs_uri: str,
    *,
    compression: str = "zstd",
    storage_options: Mapping[str, Any] | None = None,
    confirm: Literal["none", "stat"] = "stat",
    **kwargs: Any,
) -> Mapping[str, Any]:
    """
    Write a pandas or polars DataFrame directly to GCS as a single Parquet object.

    Requirements:
      - pandas path: `pyarrow` + `gcsfs` must be installed.
      - polars path: `fsspec` + `gcsfs` must be installed.

    Args:
      df: pandas.DataFrame or polars.DataFrame
      gs_uri: Destination like "gs://bucket/path/to/file.parquet"
      compression: Parquet compression {"snappy" | "zstd" | "gzip"}
      storage_options: Passed to fsspec/gcsfs (e.g., {"token": "cloud"})
      confirm: "none" to skip post-write stat, "stat" to validate and return metadata.
      **kwargs: Forwarded to the writer (e.g., pandas to_parquet or polars write_parquet)

    Returns:
      Mapping with at least {"uri": gs_uri}. If `confirm="stat"`, also includes
            {"size", "generation", "crc32c", "etag", "updated"} when available.

    Raises:
      RuntimeError: If required packages are missing.
      TypeError: If `df` isn't a pandas or polars DataFrame.
      IOError: If confirmation indicates size=0 after write.
    """
    _check_storage_uri_str(gs_uri, uri_prefix="gs://")

    storage_options = dict(storage_options or {})

    if _is_pandas_df(df):
        try:
            import pyarrow
            import gcsfs
        except ImportError as e:
            raise RuntimeError(
                "pandas write .parquet to GCS requires 'pyarrow' and 'gcsfs'"
            ) from e

        df.to_parquet(
            gs_uri,
            engine="pyarrow",
            compression=compression,
            storage_options=storage_options,
            **kwargs,
        )

    elif _is_polars_df(df):
        try:
            import fsspec
            import gcsfs
        except ImportError as e:
            raise RuntimeError(
                "polars write .parquet to GCS requires 'fsspec' and 'gcsfs'"
            ) from e

        with fsspec.open(gs_uri, "wb", **storage_options) as f:
            df.write_parquet(f, compression=compression, **kwargs)

    else:
        raise TypeError(
            f"Unsupported df type: {type(df).__name__} (expect pandas or polars)"
        )

    if confirm == "none":
        return {"uri": gs_uri}

    info = _check_gcs_file_stats(gs_uri, storage_options=storage_options)
    if int(info.get("size", 0)) <= 0:
        raise IOError(f"GCS object exists but has size=0: {gs_uri}")

    # Normalize return payload to a simple dict
    result = {
        "uri": gs_uri,
        "size": int(info.get("size")) if "size" in info else None,
        "generation": info.get("generation"),
        "crc32c": info.get("crc32c"),
        "etag": info.get("etag"),
        "updated": info.get("updated"),
    }

    return result
