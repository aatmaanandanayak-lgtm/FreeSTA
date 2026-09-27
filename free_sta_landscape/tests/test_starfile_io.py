import pandas as pd

from sta_landscape.starfile_io import particle_count, read_star, write_star


def test_roundtrip_and_count(tmp_path):
    p = tmp_path / "x.star"
    df = pd.DataFrame({"rlnClassNumber": [1, 2, 2, 3], "rlnTomoName": ["a", "a", "b", "b"]})
    write_star(str(p), {"general": {"rlnFinalResolution": 12.5}, "particles": df})
    d = read_star(str(p))
    assert d["general"]["rlnFinalResolution"] == 12.5
    assert list(d["particles"]["rlnClassNumber"]) == [1, 2, 2, 3]
    assert particle_count(str(p)) == 4


def test_column_subset_and_quotes(tmp_path):
    p = tmp_path / "y.star"
    p.write_text("\n# version 50001\n\ndata_particles\n\nloop_\n_rlnA #1\n_rlnB #2\n1 \"a b\"\n2 c\n\n")
    d = read_star(str(p), columns={"particles": ["rlnB"]})
    assert list(d["particles"].columns) == ["rlnB"]
    assert d["particles"]["rlnB"].iloc[0] == "a b"
