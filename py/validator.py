"""Validator class."""
import pandas as pd


class Validator:
    """Validator class.

    It checks a dataset for possible errors
    and data inconsistencies.

    Attributes
    ----------
    data
        Data frame to validate.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """Initialization method."""
        self.data = data

    @property
    def summary(self) -> pd.DataFrame:
        """Summary frame."""
        return pd.DataFrame({
            "idx":            [self.idx],
            "fname":          [self.fname],
            "n_records":      [self.n_records],
            "n_begin":        [self.n_begin],
            "n_end":          [self.n_end],
            "f_before":       [self.f_before],
            "f_after":        [self.f_after],
            "f_default":      [self.f_default],
            "f_expr":         [self.f_expr],
            "f_inconsistent": [self.f_inconsistent]
        }).assign(
            f_total=lambda df: df[[
                "f_before", "f_after",
                "f_default", "f_expr"
            ]].sum(axis=1),
            is_malformed=lambda df: (df["n_begin"] != 1) | (df["n_end"] != 1)
        )

    @property
    def idx(self) -> int:
        return self.data["idx"].values[0]

    @property
    def fname(self) -> str:
        return self.data["fname"].values[0]

    @property
    def n_records(self) -> int:
        return len(self.data)

    @property
    def n_before(self) -> int:
        begin_ts = self.data.loc[self.data["note"] == "BEGIN", "start"].values[0]
        return len(self.data[self.data["start"] < begin_ts])
    @property
    def f_before(self) -> float:
        return self.n_before / self.n_records

    @property
    def n_after(self) -> int:
        end_ts = self.data.loc[self.data["note"] == "END", "end"].values[0]
        return len(self.data[self.data["end"] > end_ts]) / self.n_records
    @property
    def f_after(self) -> float:
        return self.n_after / self.n_records

    @property
    def n_default(self) -> int:
        return len(self.data[self.data["tier"] == "default"])
    @property
    def f_default(self) -> float:
        return self.n_default / self.n_records

    @property
    def n_expr(self) -> int:
        return len(self.data[self.data["tier"] == "expr"])
    @property
    def f_expr(self) -> float:
        return self.n_expr / self.n_records

    @property
    def n_inconsistent(self) -> int:
        return (self.data["end"] <= self.data["start"]).sum()
    @property
    def f_inconsistent(self) -> float:
        return self.n_inconsistent / self.n_records

    @property
    def n_begin(self) -> int:
        return len(self.data[self.data["note"] == "BEGIN"])

    @property
    def n_end(self) -> int:
        return len(self.data[self.data["note"] == "END"])
