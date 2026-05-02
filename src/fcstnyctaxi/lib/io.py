from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, NamedTuple

from tsbricks.blocks.dataio import read_sql, render_sql_template


class PreparedSql(NamedTuple):
    sql_text: str
    sha256: str


def prepare_sql(sql_path: Path, sql_params: Mapping[str, object]) -> PreparedSql:
    text = read_sql(sql_path=sql_path)
    text = render_sql_template(text, sql_params)
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PreparedSql(sql_text=text, sha256=sha256)
