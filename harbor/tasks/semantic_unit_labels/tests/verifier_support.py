"""Deterministic artifact/replay helpers, shared by this task's verifier.

Copied into each task so the packaged environment has no external dependencies.
Content expectations live as independently derived literals in test_state.py.
This is not a visual-legibility checker or a hostile-code security sandbox.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile

from PIL import Image
import pytest

APP = Path("/app")
TESTS = Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path):
    assert path.is_file() and not path.is_symlink(), (
        f"required regular file is missing: {path.name}"
    )


def _figure_names(directory, *, replay=False):
    allowed = {"figure.png"}
    if replay:
        allowed.add("figure.manifest.json")
    extras = sorted(path.name for path in directory.glob("figure.*") if path.name not in allowed)
    assert not extras, f"another figure.* file conflicts with the requested PNG: {extras}"


def _pairs(items):
    result = {}
    for key, value in items:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def _nonfinite(value):
    raise AssertionError(f"JSON must not contain {value}")


def _json(path):
    _regular(path)
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_nonfinite,
    )


def _png(path):
    _regular(path)
    assert path.stat().st_size > 1000, "PNG is empty or implausibly small"
    with Image.open(path) as image:
        assert image.format == "PNG", "figure.png must actually be PNG"
        image.verify()
    with Image.open(path) as image:
        image.load()
        assert min(image.size) >= 100, "PNG has an implausibly small canvas"
        # More than four colours rules out empty/plain images, but does not
        # measure text size, contrast, occlusion, or collisions.
        assert image.convert("RGB").getcolors(maxcolors=4) is None, (
            "figure.png appears blank"
        )


@pytest.fixture(scope="module")
def submission(request):
    """Tie the delivered PNG and sidecar to a fresh captured plotting run."""
    csv_name = request.module.CSV_NAME
    input_hash = request.module.EXPECTED_INPUT_SHA256
    source_input = APP / csv_name
    _regular(source_input)
    assert _sha(source_input) == input_hash, "the supplied CSV was modified"
    _regular(APP / "plot.py")
    _figure_names(APP)
    _png(APP / "figure.png")
    delivered_png_hash = _sha(APP / "figure.png")
    delivered_sidecar = _json(APP / "plotted_values.json")

    with tempfile.TemporaryDirectory(prefix="hw2-verifier-") as temp:
        root = Path(temp)
        work = root / "workspace"
        hooks = root / "hooks"
        shutil.copytree(APP, work, symlinks=True)
        hooks.mkdir()
        # Only the capture modules enter the submitted process's import path.
        # The checks and their expected values are not imported by plot.py.
        for name in ("figure_manifest.py", "sitecustomize.py"):
            shutil.copy2(TESTS / name, hooks / name)
        for name in ("figure.png", "plotted_values.json", "figure.manifest.json"):
            path = work / name
            if path.is_file() or path.is_symlink():
                path.unlink()
            assert not path.exists(), f"output path is not a file: {name}"

        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(hooks),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "SOURCE_DATE_EPOCH": "1700000000",
        })
        # Disk-backed logs avoid pipe hangs from child/background processes.
        with (root / "stdout.log").open("wb") as out, (
            root / "stderr.log"
        ).open("wb") as err:
            process = subprocess.Popen(
                ["/usr/local/bin/python3", "plot.py"],
                cwd=work,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also stop children left behind by an otherwise exited script.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        diagnostics = (
            (root / "stdout.log").read_text(errors="replace")[-1500:]
            + (root / "stderr.log").read_text(errors="replace")[-2500:]
        )
        assert not timed_out, "plot.py replay exceeded 60 seconds\n" + diagnostics
        assert process.returncode == 0, "plot.py replay failed\n" + diagnostics

        _figure_names(work, replay=True)
        _png(work / "figure.png")
        replay_hash = _sha(work / "figure.png")
        replay_sidecar = _json(work / "plotted_values.json")
        manifest = _json(work / "figure.manifest.json")
        assert isinstance(manifest, dict), "capture manifest must be an object"
        assert manifest.get("image_sha256") == replay_hash, (
            "the saved image was overwritten after the captured Matplotlib save"
        )
        assert replay_hash == delivered_png_hash, (
            "plot.py does not reproduce the delivered figure.png byte-for-byte"
        )
        assert replay_sidecar == delivered_sidecar, (
            "plot.py does not reproduce the delivered plotted_values.json"
        )
        _regular(work / csv_name)
        assert _sha(work / csv_name) == input_hash, "replay modified its CSV"
        assert _sha(source_input) == input_hash, "replay modified the original CSV"

    return {"manifest": manifest, "sidecar": delivered_sidecar}


def _number(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"expected a JSON number, got {value!r}"
    )
    assert math.isfinite(value), f"non-finite number: {value!r}"
    return float(value)


def assert_numbers(actual, expected):
    assert isinstance(actual, list), "numeric series must be an array"
    assert len(actual) == len(expected), "wrong number of plotted observations"
    for got, want in zip(actual, expected):
        assert math.isclose(_number(got), _number(want), rel_tol=1e-8, abs_tol=1e-8), (
            f"expected {want}, got {got}"
        )


def assert_sidecar(actual, expected):
    assert isinstance(actual, dict), "sidecar must be a JSON object"
    assert set(actual) == set(expected), "sidecar keys do not match the required schema"
    for key, wanted in expected.items():
        got = actual[key]
        if isinstance(wanted, list):
            assert isinstance(got, list) and len(got) == len(wanted), (
                f"wrong array shape for {key}"
            )
            if wanted and isinstance(wanted[0], str):
                assert all(isinstance(item, str) for item in got)
                assert got == wanted, f"wrong categories/order for {key}"
            else:
                assert_numbers(got, wanted)
        else:
            assert math.isclose(_number(got), _number(wanted), rel_tol=1e-8, abs_tol=1e-8), (
                f"{key}: expected {wanted}, got {got}"
            )


def main_axes(manifest, count):
    assert manifest["n_axes"] == count, "wrong number of chart panels"
    assert manifest["n_axes_raw"] == count, "extra axes/colorbar/twin axis"
    assert manifest["n_child_axes"] == 0, "unexpected inset/secondary axes"
    axes = manifest["axes"]
    assert len(axes) == count and all(ax["role"] == "main" for ax in axes)
    assert all(ax["projection"] == "rectilinear" for ax in axes)
    # Scale restrictions belong to the task-specific checks, not to all tasks.
    return axes


def bar_series(ax):
    """Read vertical bars by position, independently of the number of bar calls."""
    bars = [item for item in ax["containers"] if item["type"] == "bar"]
    assert bars, "expected a vertical bar series"
    assert all(item["orientation"] == "vertical" for item in bars), (
        "bars must be vertical"
    )
    assert not ax["images"] and not ax["collections"], (
        "unexpected additional plotted series"
    )
    # A zero guide is a normal way to show the baseline, not an extra data series.
    # Do not broadly ignore lines: nonzero guides or lines with markers still fail.
    for line in ax["lines"]:
        assert (
            line["type"] == "line"
            and len(line["y"]) >= 2
            and all(math.isclose(_number(y), 0, abs_tol=1e-8) for y in line["y"])
            and line["marker"].casefold() in {"", "none", " "}
            and line["linestyle"].casefold() not in {"", "none", " "}
        ), "unexpected additional plotted series"

    observations = []
    for item in bars:
        values, offsets, baselines = (
            item["values"], item["offsets"], item["baselines"]
        )
        assert all(isinstance(series, list) for series in (values, offsets, baselines))
        assert len(values) == len(offsets) == len(baselines), "invalid bar capture"
        observations.extend(
            (_number(offset), _number(value), _number(baseline))
            for offset, value, baseline in zip(offsets, values, baselines, strict=True)
        )
    observations.sort(key=lambda item: item[0])
    assert observations, "expected nonempty vertical bars"
    assert all(a[0] < b[0] for a, b in zip(observations, observations[1:])), (
        "each category must have a separate bar position"
    )
    result = {
        "type": "bar",
        "orientation": "vertical",
        "offsets": [item[0] for item in observations],
        "values": [item[1] for item in observations],
        "baselines": [item[2] for item in observations],
    }
    assert_numbers(result["baselines"], [0] * len(result["values"]))
    return result


def line_series(ax):
    assert len(ax["lines"]) == 1, "expected a single connected line"
    assert not ax["containers"] and not ax["images"] and not ax["collections"]
    line = ax["lines"][0]
    assert line["type"] == "line"
    assert line["linestyle"].lower() not in {"", "none", " "}
    assert _number(line["linewidth"]) > 0
    assert line["marker"] == "o", "circular markers are required"
    return line


def assert_bar_annotations(ax, strings):
    actual = Counter(item["text"] for item in ax["annotations"])
    required = Counter(strings)
    assert not (required - actual), (
        f"missing requested value annotations: {list((required - actual).elements())}"
    )
