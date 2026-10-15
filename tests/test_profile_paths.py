"""Check profile discovery for small indices and restarted runs."""

import pytest

from mkipp2.mesa_data import get_profile_paths


def test_single_entry(tmp_path):
    """Return the indexed profile even when it is the only entry."""
    index = tmp_path / "profiles.index"
    index.write_text("1 models\n10 2 7\n")
    assert get_profile_paths([str(tmp_path)]) == [f"{tmp_path}/profile7.data"]


@pytest.mark.parametrize("contents", ["0 models\n", "", "0 models\n\n# no profiles\n"])
@pytest.mark.filterwarnings("ignore:loadtxt:UserWarning")
def test_empty_index(tmp_path, contents):
    """Reject indices without data and identify the offending file."""
    index = tmp_path / "profiles.index"
    index.write_text(contents)
    with pytest.raises(ValueError, match="contains no profile entries") as caught:
        get_profile_paths([str(tmp_path)])
    assert str(index) in str(caught.value)


def test_restart_and_directory_order(tmp_path):
    """Keep cleanup local to each directory and retain surviving order."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "profiles.index").write_text(
        "5 models\n10 2 1\n20 2 2\n30 2 3\n20 2 4\n25 2 5\n")
    (second / "profiles.index").write_text("1 models\n5 2 9\n")
    assert get_profile_paths([str(first), str(second)]) == [
        f"{first}/profile1.data", f"{first}/profile4.data",
        f"{first}/profile5.data", f"{second}/profile9.data",
    ]
