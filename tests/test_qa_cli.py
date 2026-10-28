"""CLI behavior: a missing or incomplete vectorstore fails with a clear
message, an invalid --k is rejected before anything else runs, and an
empty question is rejected before the index is loaded.

No model downloads involved: all guards run before anything is loaded.
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
    (store_dir / "index.faiss").touch()
    (store_dir / "index.pkl").touch()
    monkeypatch.setattr(sys, "argv", _argv(str(store_dir)))

    def _boom(_path):
        raise AssertionError("guard should have passed and load attempted")

    monkeypatch.setattr(qa_module, "load_vectorstore", _boom)

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


def test_vectorstore_path_is_file_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """A file (not a directory) at --vectorstore: exit(1), no FAISS traceback."""
    store_file = tmp_path / "vectorstore"
    store_file.touch()
    monkeypatch.setattr(sys, "argv", _argv(str(store_file)))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not a directory" in err
    assert "build_index" in err


@pytest.mark.parametrize("present", [[], ["index.faiss"]])
def test_incomplete_vectorstore_dir_exits_1_with_hint(tmp_path, monkeypatch, capsys, present):
    """Empty dir / dir missing index.pkl: exit(1) naming the missing file(s)."""
    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    for name in present:
        (store_dir / name).touch()
    monkeypatch.setattr(sys, "argv", _argv(str(store_dir)))

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "incomplete" in err
    assert "build_index" in err
    assert "Traceback" not in err


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
    (store_dir / "index.faiss").touch()
    (store_dir / "index.pkl").touch()
    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate", "--k", "1",
                                      "--vectorstore", str(store_dir),
                                      "what is rag?"])

    def _boom(_path):
        raise AssertionError("guard should have passed and load attempted")

    monkeypatch.setattr(qa_module, "load_vectorstore", _boom)

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


def test_interactive_eof_exits_0_without_traceback(monkeypatch, capsys):
    """Ctrl+D at the prompt: exit(0) with a short note, no traceback."""
    import src.qa as qa_module

    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate"])

    def _eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    err = capsys.readouterr().err
    assert "No question asked" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_question_exits_1_with_hint(monkeypatch, capsys, blank):
    """Empty / whitespace-only question: exit(1) before the index is loaded."""
    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate"])

    def _blank(_prompt=""):
        return blank

    monkeypatch.setattr("builtins.input", _blank)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "question is empty" in err
    assert "Traceback" not in err


def test_nonblank_question_passes_empty_guard(tmp_path, monkeypatch):
    """A real question must not trip the empty-question guard.

    Stop right after the guard by monkeypatching load_vectorstore, so no
    model download is attempted.
    """
    import src.qa as qa_module

    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    (store_dir / "index.faiss").touch()
    (store_dir / "index.pkl").touch()
    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate",
                                      "--vectorstore", str(store_dir),
                                      "what is rag?"])

    def _boom(_path):
        raise AssertionError("guard should have passed and load attempted")

    monkeypatch.setattr(qa_module, "load_vectorstore", _boom)

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


def test_interactive_keyboard_interrupt_exits_0_without_traceback(monkeypatch, capsys):
    """Ctrl+C at the prompt: exit(0) with a short note, no traceback."""
    monkeypatch.setattr(sys, "argv", ["qa", "--no-generate"])

    def _interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    err = capsys.readouterr().err
    assert "No question asked" in err
    assert "Traceback" not in err
