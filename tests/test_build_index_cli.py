"""CLI behavior: build_index fails clearly when there is nothing to index.

No model downloads involved: the guards run before embedding is attempted.
"""

import sys

import pytest

from src.build_index import main


def _argv(data_dir: str, out_dir: str, chunk_size: int | None = None,
         chunk_overlap: int | None = None) -> list:
    argv = [
        "build_index",
        "--data", data_dir,
        "--out", out_dir,
    ]
    if chunk_size is not None:
        argv += ["--chunk-size", str(chunk_size)]
    if chunk_overlap is not None:
        argv += ["--chunk-overlap", str(chunk_overlap)]
    return argv


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


def test_blank_txt_docs_exit_1_with_hint(tmp_path, monkeypatch, capsys):
    """Only blank/whitespace .txt files: exit(1) instead of a FAISS error.

    The files exist, so the earlier 'no .txt documents' guard passes, but
    chunking them yields zero chunks and FAISS cannot index an empty list.
    """
    data = tmp_path / "data"
    data.mkdir()
    (data / "blank.txt").write_text("", encoding="utf-8")
    (data / "spaces.txt").write_text("   \n\n  ", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(str(data), str(tmp_path / "out")))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "no usable text" in err
    assert str(data) in err


def test_blank_docs_mixed_with_valid_do_not_fire(
    tmp_path, monkeypatch, capsys
):
    """One blank file next to a real doc still reaches indexing.

    Monkeypatch build_vectorstore so no model download is attempted.
    """
    import src.build_index as build_index_module

    data = tmp_path / "data"
    data.mkdir()
    (data / "blank.txt").write_text("", encoding="utf-8")
    (data / "real.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(str(data), str(tmp_path / "out")))

    called = {}

    def _fake_build(chunks, out):
        called["chunks"] = len(chunks)

    monkeypatch.setattr(build_index_module, "build_vectorstore", _fake_build)
    main()
    assert called["chunks"] == 1
    assert capsys.readouterr().err == ""


def test_chunk_overlap_not_smaller_than_size_exits_1(
    tmp_path, monkeypatch, capsys
):
    """--chunk-overlap >= --chunk-size: exit(1) instead of a stuck splitter."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _argv(str(data), str(tmp_path / "out"),
                           chunk_size=400, chunk_overlap=500)
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--chunk-overlap" in err
    assert "--chunk-size" in err


def test_chunk_size_zero_exits_1(tmp_path, monkeypatch, capsys):
    """--chunk-size 0: exit(1) with a hint, not a cryptic splitter error."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _argv(str(data), str(tmp_path / "out"), chunk_size=0)
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert "--chunk-size" in capsys.readouterr().err


def test_negative_chunk_overlap_exits_1(tmp_path, monkeypatch, capsys):
    """--chunk-overlap -10: exit(1) with a hint instead of deep failure."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _argv(str(data), str(tmp_path / "out"), chunk_overlap=-10)
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert "--chunk-overlap" in capsys.readouterr().err


def test_chunk_guards_do_not_fire_on_valid_values(
    tmp_path, monkeypatch, capsys
):
    """Explicit valid --chunk-size/--chunk-overlap still reaches indexing.

    Monkeypatch build_vectorstore so no model download is attempted.
    """
    import src.build_index as build_index_module

    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("some content", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", _argv(str(data), str(tmp_path / "out"),
                           chunk_size=200, chunk_overlap=20)
    )

    called = {}

    def _fake_build(chunks, out):
        called["chunks"] = len(chunks)

    monkeypatch.setattr(build_index_module, "build_vectorstore", _fake_build)
    main()
    assert called["chunks"] == 1
    assert capsys.readouterr().err == ""
