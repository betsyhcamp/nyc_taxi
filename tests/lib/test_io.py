from pathlib import Path

import fsspec
import pytest
from fsspec.implementations.local import LocalFileSystem

from fcstnyctaxi.lib.io import (
    build_run_prefix,
    build_run_scoped_uri,
    download_from_gcs,
    prepare_sql,
    sync_to_gcs,
    upload_to_gcs,
    write_text_to_gcs,
)

EXPECTED_HASH = "e004ebd5b5532a4b85984a62f8ad48a81aa3460c1ca07701f386135d72cdecf5"


@pytest.fixture
def sql_file(tmp_path: Path) -> Path:
    """A SQL file with no Jinja placeholders."""
    path = tmp_path / "query.sql"
    path.write_text("SELECT pickup_date FROM `proj.dataset.table`")
    return path


@pytest.fixture
def sql_file_with_params(tmp_path: Path) -> Path:
    """A SQL file with Jinja placeholders."""
    path = tmp_path / "query_with_params.sql"
    path.write_text("SELECT * FROM `proj.schema.table` WHERE year = {{ year }}")
    return path


@pytest.fixture
def fake_gcs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve gs:// URIs onto a LocalFileSystem under tmp_path; return its root.

    Runs the transfer functions' real fsspec calls in CI with no credentials.
    Two details are load-bearing: auto_mkdir=True *is* the implicit-parents
    clause, since without it a put into an absent prefix raises FileNotFoundError
    where GCS would not; and the path is built by string substitution because
    LocalFileSystem._strip_protocol strips the trailing "/" that gcsfs preserves,
    and that "/" is what tells fsspec the destination is a prefix rather than the
    name of the object to write.
    """
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    fs = LocalFileSystem(auto_mkdir=True)

    def _fake_url_to_fs(url: str, **kwargs: object) -> tuple[LocalFileSystem, str]:
        return fs, url.replace("gs://", f"{remote_root}/", 1)

    monkeypatch.setattr("fsspec.url_to_fs", _fake_url_to_fs)
    return remote_root


@pytest.fixture
def download_dir(tmp_path: Path) -> Path:
    """A destination path that deliberately does not exist yet."""
    return tmp_path / "scratch" / "inputs"


@pytest.fixture
def composed_dir(tmp_path: Path) -> Path:
    """A step's local output directory: three files, one of them nested."""
    path = tmp_path / "scratch" / "compose_configs"
    (path / "nested").mkdir(parents=True)
    (path / "manifest.json").write_text("{}")
    (path / "composed_config_naive.yaml").write_text("naive")
    (path / "nested" / "extra.yaml").write_text("extra")
    return path


# ================================================
# prepare_sql tests
# ================================================


def test_prep_sql_with_param_rendering(sql_file_with_params: Path) -> None:
    """Test that prepare_sql will render sql with parameters"""
    sql_path = sql_file_with_params
    params = {"year": 1980}

    result = prepare_sql(sql_path=sql_path, sql_params=params)
    assert result.sql_text == "SELECT * FROM `proj.schema.table` WHERE year = 1980"


def test_prep_sql_raises_on_extra_param(sql_file: Path) -> None:
    "Test that prepare_sql raises ValueError if extra params supplied not in template."
    sql_path = sql_file
    params = {"foo": 1}

    with pytest.raises(ValueError, match="unused"):
        prepare_sql(sql_path=sql_path, sql_params=params)


def test_prep_sql_raises_on_missing_param(sql_file_with_params: Path) -> None:
    "Test that prepare_sql raises ValueError if param in template missing from params"
    sql_path = sql_file_with_params
    params: dict = {}

    with pytest.raises(ValueError, match="missing"):
        prepare_sql(sql_path=sql_path, sql_params=params)


def test_prep_sql_raises_filenotfound_on_missing_file(tmp_path: Path) -> None:
    """Test that nonexistant SQL file raises FileNotFoundError"""

    missing_file = tmp_path / "nonexistent_file.sql"

    with pytest.raises(FileNotFoundError):
        prepare_sql(sql_path=missing_file, sql_params={})


def test_prep_sql_hash_deterministic(tmp_path: Path) -> None:
    """Test that prepare_sql returns deterministic SHA-256 hashes."""
    sql_path = tmp_path / "q.sql"
    sql_path.write_text("SELECT 1")
    result = prepare_sql(sql_path=sql_path, sql_params={})
    assert result.sha256 == EXPECTED_HASH

    result2 = prepare_sql(sql_path=sql_path, sql_params={})
    assert result.sha256 == result2.sha256


# ================================================
# build_run_scoped_uri tests
# ================================================


def test_build_run_scoped_id_constructs_expected_string() -> None:
    bucket = "BUCKET"
    prefix = "PREFIX"
    run_id = "RUNID"
    filename = "FILE.parquet"
    uri = build_run_scoped_uri(
        bucket=bucket, prefix=prefix, run_id=run_id, filename=filename
    )
    assert uri == "gs://BUCKET/PREFIX/RUNID/FILE.parquet"


# ================================================
# build_run_prefix tests
# ================================================


def test_build_run_prefix_constructs_expected_string() -> None:
    """Test that build_run_prefix writes the convention and returns a run root.

    The trailing "/" is load-bearing: upload_to_gcs rejects a destination
    without one, so the prefix must compose with a step name by concatenation.
    """
    prefix = build_run_prefix(
        bucket="BUCKET", env="dev", slice_name="train", run_id="RUNID"
    )
    assert prefix == "gs://BUCKET/dev/train/RUNID/"


def test_build_run_prefix_raises_on_unknown_slice() -> None:
    """Test that a slice token outside SliceName raises instead of building a path."""
    with pytest.raises(ValueError, match="slice_name"):
        build_run_prefix(
            bucket="BUCKET",
            env="dev",
            slice_name="training",  # type: ignore[arg-type]
            run_id="RUNID",
        )


# ================================================
# write_text_to_gcs tests
# ================================================


def test_write_text_to_gcs_writes_text_at_uri() -> None:
    """Test that text file can be written to & read from fsspec memory system"""

    uri = "memory://text_write_to_gcs/file.sql"

    write_text_to_gcs("SELECT 1", uri)

    fs, path = fsspec.url_to_fs(uri)
    with fs.open(path, "r") as f:
        text = f.read()

    assert text == "SELECT 1"


# ================================================
# download_from_gcs tests
# ================================================


@pytest.mark.parametrize(
    "uri",
    [
        "s3://BUCKET/inputs/panel.parquet",
        "file:///tmp/panel.parquet",
        "/tmp/panel.parquet",
    ],
)
def test_download_from_gcs_rejects_non_gcs_uri(download_dir: Path, uri: str) -> None:
    """Test that a non-gs:// URI raises rather than resolving to another backend.

    Without the guard, file:// and a bare path both resolve to LocalFileSystem
    and copy the bytes successfully -- a wrong-provenance download, no error.
    """
    with pytest.raises(ValueError, match="must be a gs://"):
        download_from_gcs(uri, download_dir)

    assert not download_dir.exists()


def test_download_from_gcs_returns_the_local_path_it_wrote(
    fake_gcs: Path, download_dir: Path
) -> None:
    """Test that the object lands at destination_dir / the URI's final segment.

    destination_dir is created if missing. The impl is handed this Path
    directly, so a wrong return value misdirects the caller instead of raising.
    """
    remote = fake_gcs / "BUCKET" / "inputs"
    remote.mkdir(parents=True)
    (remote / "panel.parquet").write_text("PANEL")

    result = download_from_gcs("gs://BUCKET/inputs/panel.parquet", download_dir)

    assert result == download_dir / "panel.parquet"
    assert result.read_text() == "PANEL"


def test_download_from_gcs_raises_on_a_prefix_uri(
    fake_gcs: Path, download_dir: Path
) -> None:
    """Test that a URI naming a prefix raises rather than downloading nothing.

    download_from_gcs deliberately handles a single object only. Inbound
    directory support needs prefix listing and relative-path reconstruction;
    add it -- and drop this test -- when a caller first needs a directory.
    """
    prefix = fake_gcs / "BUCKET" / "dev" / "train" / "RUNID" / "compose_configs"
    prefix.mkdir(parents=True)
    (prefix / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="single object"):
        download_from_gcs("gs://BUCKET/dev/train/RUNID/compose_configs", download_dir)

    assert not download_dir.exists()


# ================================================
# upload_to_gcs tests
# ================================================

PREFIX_URI = "gs://BUCKET/dev/train/RUNID/compose_configs/"


@pytest.fixture
def uploaded_prefix(fake_gcs: Path) -> Path:
    """Where PREFIX_URI resolves to under the fake remote root."""
    return fake_gcs / PREFIX_URI.removeprefix("gs://")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://BUCKET/dev/train/RUNID/compose_configs/",
        "file:///tmp/compose_configs/",
        "/tmp/compose_configs/",
    ],
)
def test_upload_to_gcs_rejects_non_gcs_uri(composed_dir: Path, uri: str) -> None:
    """Test that a non-gs:// destination raises before anything is transferred.

    Every case ends in "/" so that only the scheme guard can raise; a URI
    missing the slash would trip the prefix check and pass for the wrong reason.
    """
    with pytest.raises(ValueError, match="must be a gs://"):
        upload_to_gcs(composed_dir, uri)


def test_upload_to_gcs_places_directory_contents_under_prefix(
    uploaded_prefix: Path, composed_dir: Path
) -> None:
    """Test that a directory's contents land under the prefix, basename dropped.

    Repeating the basename would give .../compose_configs/compose_configs/, which
    raises nothing and leaves every artifact one level below where readers look.
    """
    upload_to_gcs(composed_dir, PREFIX_URI)

    landed = {
        str(p.relative_to(uploaded_prefix))
        for p in uploaded_prefix.rglob("*")
        if p.is_file()
    }
    source = {
        str(p.relative_to(composed_dir)) for p in composed_dir.rglob("*") if p.is_file()
    }
    assert landed == source
    assert not (uploaded_prefix / "compose_configs").exists()


def test_upload_to_gcs_rejects_destination_without_trailing_slash(
    fake_gcs: Path, composed_dir: Path
) -> None:
    """Test that a destination prefix missing its trailing "/" raises, writing nothing.

    Rejected rather than normalised because the mistake is otherwise invisible:
    every object is written and a plausible count comes back. fsspec forgives the
    omission once the prefix exists, so only a fresh run_id would surface the
    mistake.
    """
    with pytest.raises(ValueError, match="ending in"):
        upload_to_gcs(composed_dir, PREFIX_URI.rstrip("/"))

    assert list(fake_gcs.rglob("*")) == []


def test_upload_to_gcs_overwrites_existing_objects(
    uploaded_prefix: Path, composed_dir: Path
) -> None:
    """Test that a second upload replaces the objects the first one wrote.

    Retrying with the same run_id must neither need a manual delete nor keep the
    earlier attempt's bytes.
    """
    upload_to_gcs(composed_dir, PREFIX_URI)
    (composed_dir / "manifest.json").write_text('{"attempt": 2}')
    upload_to_gcs(composed_dir, PREFIX_URI)

    assert (uploaded_prefix / "manifest.json").read_text() == '{"attempt": 2}'


def test_upload_to_gcs_returns_one_for_a_file(
    uploaded_prefix: Path, tmp_path: Path
) -> None:
    """Test that uploading a single file reports one object written.

    The destination prefix has no existing parents, which GCS treats as
    implicit; a local filesystem would raise FileNotFoundError instead.
    """
    source = tmp_path / "scratch" / "manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")

    assert upload_to_gcs(source, PREFIX_URI) == 1
    assert (uploaded_prefix / "manifest.json").read_text() == "{}"


def test_upload_to_gcs_returns_file_count_for_a_directory(
    fake_gcs: Path, composed_dir: Path
) -> None:
    """Test that a directory upload counts objects, not filesystem entries.

    composed_dir holds three files and one subdirectory. fsspec's own transfer
    callback counts the directory entries too and would answer five, so the count
    cannot come from the library.
    """
    assert upload_to_gcs(composed_dir, PREFIX_URI) == 3


# ================================================
# sync_to_gcs tests
# ================================================

# Sorts before every other name in sync_dir on purpose. A marker that sorted last
# would let this fake's ordered upload satisfy the ordering test for a reason that
# does not transfer to gcsfs, which uploads a batch concurrently.
MARKER = "_run_identity.json"


class _TrackingFileSystem(LocalFileSystem):
    """A fake remote recording put order, able to fail one chosen target.

    cachable is off: fsspec's cache would hand the next test this recording.
    """

    cachable = False

    def __init__(self) -> None:
        super().__init__(auto_mkdir=True)
        self.put_targets: list[str] = []
        self.fail_on: str | None = None

    def put_file(self, path1: str, path2: str, **kwargs: object) -> None:
        target = Path(path2).name
        self.put_targets.append(target)
        if target == self.fail_on:
            raise OSError(f"injected upload failure for {target}")
        super().put_file(path1, path2, **kwargs)


@pytest.fixture
def tracking_gcs(
    fake_gcs: Path, monkeypatch: pytest.MonkeyPatch
) -> _TrackingFileSystem:
    """fake_gcs, but resolving onto a filesystem that records and can fail puts."""
    fs = _TrackingFileSystem()

    def _fake_url_to_fs(url: str, **kwargs: object) -> tuple[_TrackingFileSystem, str]:
        return fs, url.replace("gs://", f"{fake_gcs}/", 1)

    monkeypatch.setattr("fsspec.url_to_fs", _fake_url_to_fs)
    return fs


@pytest.fixture
def sync_dir(tmp_path: Path) -> Path:
    """A step's output directory: two configs, a nested file, and a marker."""
    path = tmp_path / "scratch" / "compose_configs"
    (path / "nested").mkdir(parents=True)
    (path / MARKER).write_text('{"run": 1}')
    (path / "modeling.yaml").write_text("modeling")
    (path / "composed_config_naive.yaml").write_text("naive")
    (path / "nested" / "extra.yaml").write_text("extra")
    return path


@pytest.mark.parametrize(
    "uri",
    [
        "s3://BUCKET/dev/train/RUNID/compose_configs/",
        "/tmp/compose_configs/",
        PREFIX_URI.rstrip("/"),
    ],
)
def test_sync_to_gcs_rejects_a_bad_destination_before_deleting_anything(
    uploaded_prefix: Path, sync_dir: Path, uri: str
) -> None:
    """Test that an unusable destination raises with the prior run still intact.

    Where the guard sits matters here in a way it does not for upload_to_gcs,
    whose first remote act is a write. This deletes first, and a bare local path
    resolves to LocalFileSystem, so a guard left to the upload would object only
    after a real file had gone.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / MARKER).write_text('{"run": 0}')

    with pytest.raises(ValueError, match="gcs_uri must be"):
        sync_to_gcs(sync_dir, uri, completion_marker=MARKER)

    assert (uploaded_prefix / MARKER).read_text() == '{"run": 0}'


def test_sync_to_gcs_raises_rather_than_reconciling_an_absent_local_dir(
    uploaded_prefix: Path, tmp_path: Path
) -> None:
    """Test that a local_dir which does not exist raises instead of emptying the prefix.

    rglob on a missing directory yields nothing rather than raising, so without
    the guard every remote object would look unaccounted for and the reconcile
    would delete the lot.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / "modeling.yaml").write_text("modeling")

    with pytest.raises(NotADirectoryError, match="local_dir"):
        sync_to_gcs(tmp_path / "never_created", PREFIX_URI)

    assert (uploaded_prefix / "modeling.yaml").exists()


def test_sync_to_gcs_raises_before_mutating_when_the_marker_is_absent(
    uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that a completion_marker naming no local file raises, touching nothing.

    Skipping instead would not leave an obviously incomplete prefix. The
    exclusion would match nothing, so the real marker would upload inside the
    concurrent batch and a partial failure could publish it over a missing
    sibling -- the ordering off, with nothing announcing it.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / MARKER).write_text('{"run": 0}')

    with pytest.raises(FileNotFoundError, match="completion_marker"):
        sync_to_gcs(sync_dir, PREFIX_URI, completion_marker="_run_identity.jsn")

    assert (uploaded_prefix / MARKER).read_text() == '{"run": 0}'
    assert not (uploaded_prefix / "modeling.yaml").exists()


def test_sync_to_gcs_refreshes_content_and_removes_stale_objects(
    uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that the prefix ends up holding exactly local_dir, nested files included.

    The stale config is a model dropped from model_roles between two runs under
    one id: upload_to_gcs would overwrite its siblings and leave it durable, and
    a composed_config_*.yaml glob would then report a run that never happened.
    The stale nested object shares a basename with a live top-level one, so a
    reconcile comparing basenames would keep it and this would fail.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / "modeling.yaml").write_text("stale modeling")
    (uploaded_prefix / "composed_config_lightgbm.yaml").write_text("dropped model")
    (uploaded_prefix / "nested").mkdir()
    (uploaded_prefix / "nested" / "modeling.yaml").write_text("stale nested")

    uploaded, removed = sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=MARKER)

    landed = {
        str(p.relative_to(uploaded_prefix))
        for p in uploaded_prefix.rglob("*")
        if p.is_file()
    }
    source = {str(p.relative_to(sync_dir)) for p in sync_dir.rglob("*") if p.is_file()}
    assert landed == source
    assert (uploaded_prefix / "modeling.yaml").read_text() == "modeling"
    assert (uploaded, removed) == (len(source), 2)


def test_sync_to_gcs_leaves_a_prefix_with_no_extras_unchanged(
    uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that re-publishing an unchanged directory removes nothing.

    The reconcile has to recognise the objects this function itself just wrote.
    Were the relative path derived wrongly -- an unstripped root, or a basename
    comparison -- every object would look like an extra, and the second publish
    would delete the run the first one had written.
    """
    sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=MARKER)
    uploaded, removed = sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=MARKER)

    landed = [p for p in uploaded_prefix.rglob("*") if p.is_file()]
    assert removed == 0
    assert uploaded == len(landed)


def test_sync_to_gcs_uploads_the_completion_marker_last(
    tracking_gcs: _TrackingFileSystem, sync_dir: Path
) -> None:
    """Test that the marker is written after every other object, exactly once.

    A marker carried along in the batch would satisfy "present" without meaning
    "complete", since gcsfs uploads a batch concurrently. MARKER sorts first, so
    a batch upload could not produce this order by accident.
    """
    sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=MARKER)

    assert tracking_gcs.put_targets[-1] == MARKER
    assert tracking_gcs.put_targets.count(MARKER) == 1


def test_sync_to_gcs_leaves_no_marker_when_an_upload_fails(
    tracking_gcs: _TrackingFileSystem, uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that a partial upload publishes no marker and drops the previous one.

    The prefix is left holding a mix of two runs, which is what a concurrent
    batch does on failure whatever this function chooses. What it can choose is
    whether the token claiming the run is complete survives: it does not, so the
    prefix reads unfinished, which is true.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / MARKER).write_text('{"run": 0}')
    tracking_gcs.fail_on = "modeling.yaml"

    with pytest.raises(OSError, match="injected"):
        sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=MARKER)

    assert (uploaded_prefix / "composed_config_naive.yaml").read_text() == "naive"
    assert not (uploaded_prefix / MARKER).exists()


def test_sync_to_gcs_rejects_a_completion_marker_that_is_not_a_filename(
    uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that a path-shaped completion_marker raises with the prefix untouched.

    The absolute form defeats every other guard: local_dir / <absolute> discards
    local_dir, so the file resolves and is_file() passes, while the remote key
    comes from the raw string and lands nowhere near the prefix -- leaving the
    previous run's marker in place over a new run's objects.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / MARKER).write_text('{"run": 0}')
    path_shaped = (
        str(sync_dir / MARKER),
        "nested/extra.yaml",
        f"nested/../{MARKER}",
    )

    for marker_arg in path_shaped:
        with pytest.raises(ValueError, match="must be a filename"):
            sync_to_gcs(sync_dir, PREFIX_URI, completion_marker=marker_arg)

    assert (uploaded_prefix / MARKER).read_text() == '{"run": 0}'
    assert not (uploaded_prefix / "modeling.yaml").exists()


def test_sync_to_gcs_leaves_an_object_named_like_the_prefix_alone(
    fake_gcs: Path, tmp_path: Path
) -> None:
    """Test that a sibling object named exactly like the prefix is not reconciled.

    GCS has no directories, so gs://.../compose_configs and
    gs://.../compose_configs/manifest.json can both exist -- the first a sibling
    of the prefix rather than something under it. A local filesystem cannot hold
    both, so local_dir is empty here, which still exercises the reconcile.
    """
    sibling = fake_gcs / PREFIX_URI.removeprefix("gs://").rstrip("/")
    sibling.parent.mkdir(parents=True)
    sibling.write_text("an object, not a prefix")
    empty_dir = tmp_path / "scratch" / "empty"
    empty_dir.mkdir(parents=True)

    assert sync_to_gcs(empty_dir, PREFIX_URI) == (0, 0)
    assert sibling.read_text() == "an object, not a prefix"


def test_sync_to_gcs_publishes_without_a_marker_when_none_is_named(
    uploaded_prefix: Path, sync_dir: Path
) -> None:
    """Test that completion_marker=None still uploads and reconciles.

    The guarantee narrows to "the prefix holds exactly local_dir": nothing is
    deleted first, nothing is written last, and MARKER becomes an ordinary file
    with no special handling.
    """
    uploaded_prefix.mkdir(parents=True)
    (uploaded_prefix / "composed_config_lightgbm.yaml").write_text("dropped model")

    uploaded, removed = sync_to_gcs(sync_dir, PREFIX_URI)

    landed = {
        str(p.relative_to(uploaded_prefix))
        for p in uploaded_prefix.rglob("*")
        if p.is_file()
    }
    source = {str(p.relative_to(sync_dir)) for p in sync_dir.rglob("*") if p.is_file()}
    assert landed == source
    assert (uploaded, removed) == (len(source), 1)
