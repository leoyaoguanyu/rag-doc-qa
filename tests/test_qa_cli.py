"""CLI behavior: a missing vectorstore directory fails with a clear message,
and an invalid --k is rejected before anything else runs.

No model downloads involved: both guards run before anything is loaded.
"""

import sys

import pytest

from src.qa import main


def _argv(vectorstore: str) -> list:
    return ["qa", "--no-generate", "--vectorstore", vectorstore, "what is rag?"]


def _argv_with_k(k: int) -> list:
    return ["qa", "--no-generate", "--k", str(k), "what is rag?"]


def test_missing_vectorstore_dir_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """No index built yet: exit(1) and tell the user how to build one."""
    missing = str(tmp_path / "no-index-here")
    monkeypatch.setattr(sys, "argv", _argv(missing))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "build_index" in err
    assert missing in err


def test_existing_vectorstore_dir_does_not_hit_guard(tmp_path, monkeypatch):
    """Guard only fires on a missing path; an existing dir proceeds onward.

    Stop right after the guard by monkeypatching load_vectorstore, so no
    model download is attempted.
    """
    import src.qa as qa_module

    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    monkeypatch.setattr(sys, "argv", _argv(str(store_dir)))

    def _boom(_path):
        raise AssertionError("guard should have passed and load attempted")

    monkeypatch.setattr(qa_module, "load_vectorstore", _boom)

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


@pytest.mark.parametrize("bad_k", [0, -3])
def test_invalid_k_exits_1_with_hint(monkeypatch, capsys, bad_k):
    """--k 0 / negative: exit(1) with a hint, before the index is touched."""
    monkeypatch.setattr(sys, "argv", _argv_with_k(bad_k))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--k" in err
    assert str(bad_k) in err


def test_valid_k_passes_k_guard(tmp_path, monkeypatch):
    """k=1 must not trip the --k guard: execution proceeds past it.

    Stop right after the guard by monkeypatching load_vectorstore, so no
    model download is attempted.
    """
    import src.qa as qa_module

    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate", "--k", "1",
                                      "--vectorstore", str(store_dir),
                                      "what is rag?"])

    def _boom(_path):
        raise AssertionError("guard should have passed and load attempted")

    monkeypatch.setattr(qa_module, "load_vectorstore", _boom)

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()
