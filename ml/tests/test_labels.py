import numpy as np
import pandas as pd
import pytest

from cardiomamba.data.labels import (
    CLASS_NAMES,
    build_code_to_superclass,
    multi_hot,
    parse_scp_codes,
    superclasses_for,
)


@pytest.fixture
def scp_statements() -> pd.DataFrame:
    """Small excerpt shaped like scp_statements.csv (diagnostic, form and rhythm rows)."""
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
            "form": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
            "rhythm": [np.nan] * 7 + [1.0],
            "diagnostic_class": ["NORM", "MI", "MI", "STTC", "CD", "HYP", np.nan, np.nan],
        },
        index=["NORM", "IMI", "AMI", "NDT", "CLBBB", "LVH", "LVOLT", "SR"],
    )


def encode(codes: dict, scp: pd.DataFrame) -> list[int]:
    mapping = build_code_to_superclass(scp)
    return multi_hot(superclasses_for(codes, mapping, known_codes=scp.index)).astype(int).tolist()


def test_class_order_is_fixed():
    assert CLASS_NAMES == ("NORM", "MI", "STTC", "CD", "HYP")


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ({"NORM": 100.0}, [1, 0, 0, 0, 0]),
        ({"IMI": 100.0}, [0, 1, 0, 0, 0]),
        ({"NDT": 100.0}, [0, 0, 1, 0, 0]),
        ({"CLBBB": 100.0}, [0, 0, 0, 1, 0]),
        ({"LVH": 100.0}, [0, 0, 0, 0, 1]),
    ],
)
def test_single_superclass(codes, expected, scp_statements):
    assert encode(codes, scp_statements) == expected


def test_multi_label_combinations(scp_statements):
    assert encode({"IMI": 100.0, "CLBBB": 100.0}, scp_statements) == [0, 1, 0, 1, 0]
    assert encode({"NDT": 50.0, "LVH": 100.0, "CLBBB": 80.0}, scp_statements) == [0, 0, 1, 1, 1]
    # two statements in the same superclass still yield a single positive
    assert encode({"IMI": 100.0, "AMI": 100.0}, scp_statements) == [0, 1, 0, 0, 0]


def test_likelihood_is_ignored(scp_statements):
    assert encode({"IMI": 0.0}, scp_statements) == [0, 1, 0, 0, 0]
    assert encode({"IMI": 15.0}, scp_statements) == encode({"IMI": 100.0}, scp_statements)


def test_non_diagnostic_statements_do_not_contribute(scp_statements):
    assert encode({"LVOLT": 100.0, "SR": 0.0}, scp_statements) == [0, 0, 0, 0, 0]
    assert encode({"NORM": 100.0, "LVOLT": 0.0, "SR": 0.0}, scp_statements) == [1, 0, 0, 0, 0]


def test_unknown_code_raises(scp_statements):
    with pytest.raises(ValueError, match="not present"):
        encode({"NOT_A_CODE": 100.0}, scp_statements)


def test_parse_scp_codes():
    assert parse_scp_codes("{'NORM': 100.0, 'SR': 0.0}") == {"NORM": 100.0, "SR": 0.0}
    with pytest.raises(TypeError):
        parse_scp_codes("['NORM']")


def test_multi_hot_dtype_and_shape():
    y = multi_hot({"MI", "HYP"})
    assert y.dtype == np.float32 and y.shape == (5,)
