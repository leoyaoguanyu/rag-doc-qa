"""CLI behavior: an invalid --k is rejected before anything is loaded,
mirroring the guard in src.qa.main.

Also covers the --eval input-file validation: a missing / malformed
eval file is rejected with a friendly error instead of a traceback.

No model downloads involved: the guards run before the vector store
is touched.
"""

import json as json_module
import sys

import pytest

from src.evaluate import main


@pytest.mark.parametrize("bad_k", [0, -3])
def test_invalid_k_exits_1_with_hint(monkeypatch, capsys, bad_k):
    """--k 0 / negative: exit(1) with a hint, before anything is loaded."""
    monkeypatch.setattr(sys, "argv", ["evaluate", "--k", str(bad_k)])

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
    import json as json_module

    import src.evaluate as eval_module

    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json_module.dumps([{"question": "q", "expected_source": "s"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", "--k", "1",
                                      "--eval", str(eval_file),
                                      "--vectorstore", str(tmp_path)])

    class FakeStore:
        def similarity_search(self, question, k):
            raise AssertionError("guard should have passed and store attempted")

    monkeypatch.setattr(eval_module, "load_vectorstore",
                        lambda _path: FakeStore())

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


def test_missing_vectorstore_dir_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """--vectorstore pointing nowhere: exit(1) with a build_index hint.

    Mirrors the guard in src.qa.main: the missing index must fail with a
    friendly error instead of a FAISS traceback.
    """
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json_module.dumps([{"question": "q", "expected_source": "s"}]),
        encoding="utf-8",
    )
    missing = tmp_path / "nope_vectorstore"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(eval_file),
                                      "--vectorstore", str(missing)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "vectorstore" in err
    assert "build_index" in err
    assert "Traceback" not in err


def test_existing_vectorstore_dir_passes_guard(tmp_path, monkeypatch):
    """An existing --vectorstore dir must not trip the guard.

    Stop right after the guard by monkeypatching load_vectorstore, so no
    model download is attempted.
    """
    import src.evaluate as eval_module

    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json_module.dumps([{"question": "q", "expected_source": "s"}]),
        encoding="utf-8",
    )
    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(eval_file),
                                      "--vectorstore", str(store_dir)])

    class FakeStore:
        def similarity_search(self, question, k):
            raise AssertionError("guard should have passed and store attempted")

    monkeypatch.setattr(eval_module, "load_vectorstore",
                        lambda _path: FakeStore())

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()


def test_missing_eval_file_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """--eval pointing nowhere: exit(1) with a hint, no traceback."""
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(missing)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "--eval" in err
    assert str(missing) in err
    assert "Traceback" not in err


def test_invalid_json_eval_file_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """--eval with broken JSON: exit(1) naming the problem, no traceback."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(bad)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "JSON" in err
    assert "Traceback" not in err


def test_non_list_eval_file_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """--eval with a JSON object instead of a list: exit(1), no traceback."""
    bad = tmp_path / "obj.json"
    bad.write_text(json_module.dumps({"question": "q"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(bad)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "non-empty JSON list" in err
    assert "Traceback" not in err


def test_empty_pairs_list_exits_1_with_hint(tmp_path, monkeypatch, capsys):
    """--eval with an empty list: exit(1) (empty eval would crash the report)."""
    empty = tmp_path / "empty.json"
    empty.write_text(json_module.dumps([]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(empty)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "non-empty JSON list" in err
    assert "Traceback" not in err


def test_malformed_pair_exits_1_naming_pair_index(tmp_path, monkeypatch, capsys):
    """A pair missing expected_source: exit(1) pointing at the pair index."""
    bad = tmp_path / "malformed.json"
    bad.write_text(
        json_module.dumps([
            {"question": "q", "expected_source": "s"},
            {"question": "q2"},
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(bad)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "pair #1" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("bad_pair", [
    {"question": "", "expected_source": "s"},
    {"question": "   ", "expected_source": "s"},
    {"question": "q", "expected_source": ""},
    {"question": 123, "expected_source": "s"},
])
def test_blank_or_non_string_pair_exits_1_naming_pair_index(
    tmp_path, monkeypatch, capsys, bad_pair
):
    """A pair with a blank/non-string question or source: exit(1), no traceback.

    The old check only verified the keys exist, so a blank question
    slipped through and was scored against an empty retrieval.
    """
    bad = tmp_path / "blank.json"
    bad.write_text(json_module.dumps([bad_pair]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(bad)])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "pair #0" in err
    assert "non-empty" in err
    assert "Traceback" not in err


def test_valid_eval_file_passes_guard(tmp_path, monkeypatch):
    """A well-formed eval file must not trip the guard: execution proceeds past it.

    Stop right after the guard by monkeypatching load_vectorstore, so no
    model download is attempted.
    """
    import src.evaluate as eval_module

    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        json_module.dumps([{"question": "q", "expected_source": "s"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", "--eval", str(eval_file),
                                      "--vectorstore", str(tmp_path)])

    class FakeStore:
        def similarity_search(self, question, k):
            raise AssertionError("guard should have passed and store attempted")

    monkeypatch.setattr(eval_module, "load_vectorstore",
                        lambda _path: FakeStore())

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()
