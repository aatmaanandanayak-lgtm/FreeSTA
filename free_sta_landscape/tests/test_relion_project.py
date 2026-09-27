from sta_landscape.relion_project import _clean_tokens, _main_command, parse_flags


def test_parse_flags_with_empty_and_negative_values():
    toks = _clean_tokens('`which relion_refine_mpi` --o Class3D/job012/run --K 4 --gpu "" --ini_high -1 --zero_mask')
    f = parse_flags(toks)
    assert f["K"] == 4.0 and f["gpu"] == "" and f["ini_high"] == -1.0 and f["zero_mask"] is True


def test_continue_overrides():
    prog, f = _main_command([
        "`which relion_refine_mpi` --o Class3D/job012/run --iter 25 --K 4 --tau2_fudge 2",
        "`which relion_refine_mpi` --continue Class3D/job012/run_it025_optimiser.star --o Class3D/job012/run_ct25 --iter 40",
    ])
    assert prog == "relion_refine" and f["iter"] == 40.0 and f["K"] == 4.0 and "continue" not in f
