"""Uniform sequence generator."""
import numpy as np
import pandas as pd


class Uniseq:
    """Uniform sequence generator class.

    It takes raw turn-taking data and transforms it
    in a uniform sequence of bins of a given time resolution
    with information on the speakers active in each bin.

    Attributes
    ----------
    data
        Conversation data frame.
    resolution
        Time resolution (bin width) in seconds.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        resolution: float = .1
    ) -> None:
        self.data = data
        if resolution <= 0:
            raise ValueError("'resolution' has to be positive")
        self.resolution = float(resolution)

    @property
    def start(self) -> pd.Timestamp:
        return self.data["start"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.data["end"].max()

    @property
    def nbins(self) -> int:
        seconds = (self.end - self.start).total_seconds()
        return np.ceil(seconds / self.resolution).astype(int)

    # Methods -----------------------------------------------------------------

    def make_uniseq_df(self) -> pd.DataFrame:
        """Make uniform sequence data frame."""
        df = self.data
        codes, index = pd.factorize(df["tier"])
        ntiers = len(index)
        U = np.zeros((self.nbins, 1+ntiers), dtype=int)
        U[:, 0] = np.arange(self.nbins)

        C = np.column_stack([
            codes,
            (df["start"] - self.start) \
                .dt.total_seconds() \
                .div(self.resolution) \
                .astype(int),
            (df["end"] - self.start) \
                .dt.total_seconds() \
                .div(self.resolution) \
                .astype(int) \
                .add(1)
        ])

        for code, i, j in C:
            U[i:j, code+1] = 1

        udf = pd.DataFrame(U, columns=[
            "t", "tier1", "tier2", "tier3", "tier4"
        ])
        udf["n_speakers"] = udf.loc[:, "tier1":"tier4"].sum(axis=1)
        return udf

    def dump(self) -> pd.DataFrame:
        """Dump to a uniform sequence data frame."""
        df = self.data
        udf = self.make_uniseq_df()
        for col in ("condition", "fname", "idx"):
            if col in df:
                udf.insert(0, col, df[col].to_numpy()[0])
        return udf
