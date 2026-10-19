"""CLI behavior: an invalid --k is rejected before anything is loaded,
mirroring the guard in src.qa.main.

No model downloads involved: the guard runs before the eval file is read
or the vector store is touched.
"""

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
                                      "--eval", str(eval_file)])

    class FakeStore:
        def similarity_search(self, question, k):
            raise AssertionError("guard should have passed and store attempted")

    monkeypatch.setattr(eval_module, "load_vectorstore",
                        lambda _path: FakeStore())

    with pytest.raises(AssertionError, match="guard should have passed"):
        main()
