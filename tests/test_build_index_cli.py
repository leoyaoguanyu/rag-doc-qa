"""CLI behavior: build_index fails clearly when there is nothing to index.

No model downloads involved: the guards run before embedding is attempted.
"""

import sys

import pytest

from src.build_index import main


def _argv(data_dir: str, out_dir: str) -> list:
    return [
        "build_index",
        "--data", data_dir,
        "--out", out_dir,
    ]


def test_missing_data_dir_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """Wrong --data path: exit(1) and point at the expected folder content."""
    missing = str(tmp_path / "no-data-here")
    monkeypatch.setattr(sys, "argv", _argv(missing, str(tmp_path / "out")))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert missing in err
    assert "--data" in err


def test_data_dir_without_txt_docs_exits_1_with_hint(
    tmp_path, monkeypatch, capsys
):
    """Empty or .txt-less data dir: exit(1) instead of a cryptic FAISS error."""
    empty = tmp_path / "empty-data"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", _argv(str(empty), str(tmp_path / "out")))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "no .txt documents" in err
    assert str(empty) in err


def test_guards_do_not_fire_on_valid_data(tmp_path, monkeypatch):
    """Guards only fire on bad input; a valid dir proceeds to indexing.

    Stop right after the guards by monkeypatching build_vectorstore, so no
    model download is attempted.
    """
    import src.build_index as build_index_module

    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(str(data), str(tmp_path / "out")))

    called = {}

    def _fake_build(chunks, out):
        called["chunks"] = len(chunks)

    monkeypatch.setattr(build_index_module, "build_vectorstore", _fake_build)
    main()
    assert called["chunks"] == 1
