import math

from sta_landscape.codec import decode_features, encode_flags, healpix_deg, sym_order


def test_healpix_and_symmetry():
    assert healpix_deg(2) == 7.5
    assert sym_order("C1") == 1 and sym_order("C2") == 2 and sym_order("D3") == 6 and sym_order("O") == 24


def test_encode_decode_roundtrip_same_particle():
    flags = {"tau2_fudge": 4.0, "K": 5.0, "particle_diameter": 200.0, "offset_range": 5.0,
             "strict_highres_exp": 10.0, "healpix_order": 3.0, "zero_mask": True}
    f = encode_flags(flags, apix=3.4, diameter_A=180.0)
    assert math.isclose(f["particle_diameter_rel"], 200 / 180)
    assert math.isclose(f["highres_limit_rel"], 6.8 / 10.0)
    back = decode_features(f, apix=3.4, diameter_A=180.0, bool_cols={"zero_mask"}, int_cols={"K"})
    assert back["tau2_fudge"] == 4.0 and back["K"] == 5 and back["healpix_order"] == 3
    assert back["particle_diameter"] == 200 and back["strict_highres_exp"] == 10.0 and back["zero_mask"] is True


def test_transfer_to_bigger_particle_samples_finer():
    f = encode_flags({"healpix_order": 2.0, "particle_diameter": 200.0}, apix=3.4, diameter_A=150.0)
    big = decode_features(f, apix=3.4, diameter_A=600.0)
    assert big["healpix_order"] == 4          # 4x larger particle -> 4x finer angular step
    assert big["particle_diameter"] == 800
