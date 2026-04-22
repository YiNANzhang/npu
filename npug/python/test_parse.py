"""Smoke test: parse the Rust-produced golden fixture from Python."""

import sys
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "generated"))

from Npug.Graph import Graph  # noqa: E402


def test_parse_golden():
    fixture = HERE.parent / "fixtures" / "minimal_v0_1.npug"
    if not fixture.exists():
        subprocess.check_call(
            ["cargo", "run", "-p", "npug", "--example", "gen_golden"],
            cwd=HERE.parent.parent,
        )
    buf = fixture.read_bytes()
    g = Graph.GetRootAs(buf, 0)
    assert g.AbiVersion() == 0x000100, f"expected 0x000100, got {g.AbiVersion():#x}"
    assert g.Producer().decode() == "npug-golden-gen/0.1.0"
    assert g.TensorsLength() == 3
    assert g.KernelsLength() == 1
    assert g.EntryPointsLength() == 1


if __name__ == "__main__":
    test_parse_golden()
    print("OK")
