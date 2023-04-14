"""Data loader class.

This is a simple class for loading subsequent datasets.
"""
import os
from typing import Any, Optional, Iterable
from pathlib import Path
from itertools import chain
import numpy as np
import pandas as pd


PathLike = str | bytes | os.PathLike


class Dataloader:
    """Dataloader class.

    Attributes
    ----------
    path
        Path to the data directory.
    propmap
        Property map for assigning metadata to groups.
    patterns
        List of glob patterns to look for.
    dtfmt
        Datetime string format string used
        to infer experiment dates from file names.
    csv_kws
        Dictionary with other keyword arguments
        passed to :func:`pandas.read_csv`.
    **kwds
        Are merged into ``self.csv_kws``.
        Repeated keys raise :class:`TypeErrors`.
    """
    _columns = (
        "tier",
        "start",
        "end",
        "note"
    )

    def __init__(
        self,
        path: PathLike,
        propmap: Optional[PathLike] = None,
        *,
        patterns: str | Iterable[str] = "*",
        dtfmt: str = "%d.%m.%Y_%H.%M",
        csv_kws: Optional[dict] = None,
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        propmap
            Path to a propperty map file.
        """
        self.path = Path(path).absolute()
        self.propmap = propmap
        if self.propmap:
            self.propmap = pd.read_csv(propmap, sep=None, engine="python") \
                .drop_duplicates() \
                .reset_index(drop=True) \
                .set_index("group")
            self.propmap["condition"].replace({
                "t":  "creative",
                "m":  "math",
                "tk": "controv"
            }, inplace=True)
        self.patterns = (patterns,) if isinstance(patterns, str) \
            else tuple(patterns)
        self.dtfmt = dtfmt
        self.csv_kws = { **(csv_kws or {}), **kwds }

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}('{self.path}')"

    def __iter__(self) -> Iterable[pd.DataFrame]:
        yield from self.load()

    def __len__(self) -> int:
        return self.n_paths

    @property
    def n_paths(self) -> int:
        return sum(1 for _ in self.iterpaths())

    def iterpaths(self) -> Iterable[PathLike]:
        """Iterate over data paths."""
        yield from chain(*(self.path.glob(p) for p in self.patterns))

    def load(self, **kwds: Any) -> Iterable[pd.DataFrame]:
        """Load datasets and iterate over them one by one.

        Parameters
        ----------
        **kwds
            Passed to :py:meth:`read_frame`.
        """
        for idx, path in enumerate(self.iterpaths(), 1):
            df = self.read_frame(path, **kwds)
            df.insert(0, "idx", idx)
            yield df

    def read_frame(
        self,
        path: PathLike,
        *,
        raw: bool = False,
        **kwds: Any
    ) -> pd.DataFrame:
        """Read data frame from any format.

        Parameters
        ----------
        path
            Path to a data file.
        raw
            Should raw data frames (before filtering)
            be returned.
        """
        # pylint: disable=too-many-locals
        ext = path.suffix
        if ext == ".txt":
            sep = "\t"
        elif ext == ".csv":
            sep = ","
        else:
            raise TypeError(f"cannot read '{ext}' files")

        ts = pd.to_datetime(path.stem, format=self.dtfmt)
        df = pd.read_csv(path, **{ "sep": sep, **self.csv_kws, **kwds })
        empty = [ c for c in df.columns if df[c].isnull().all() ]
        df.drop(columns=empty, inplace=True)

        # Check integer columns for timestamps
        cols = []
        tests = {
            "int": pd.api.types.is_integer_dtype,
            "float": pd.api.types.is_float_dtype
        }

        for typ, test in tests.items():
            cols = [ c for c in df.columns if test(df[c]) ]
            if len(cols) > 2:
                cols = cols[:2]
                df = df.loc[:, [0, *cols, df.columns[-1]]]
                if typ == "int":
                    df[cols] /= 1000
                break

        df.columns = self._columns

        for col in ("start", "end"):
            df[col] = self.parse_time(df[col], ts)

        for col in ("tier", "note"):
            df[col] = self.sanitize_strings(df[col])
        df.loc[df["note"] == "default", "tier"] = "default"
        mask = df["tier"].str.startswith("osoba")
        df.loc[mask, "note"] = df.loc[mask, "tier"]

        if self.propmap is not None:
            df.insert(0, "condition", self.propmap.loc[path.stem, "condition"])
        df.insert(0, "ts", ts)
        df.insert(0, "fname", f"{path.stem}{path.suffix}")
        duration = (df["end"] - df["start"]).dt.total_seconds()
        df.insert(df.shape[1]-1, "duration", duration)

        if not raw:
            df = df \
                .sort_values(by=["start", "end"], ascending=True) \
                .pipe(self.filter) \
                .pipe(self.merge_contiguous) \
                .pipe(self.check_overlaps) \
                .pipe(self.postprocess)
        return df

    def parse_time(self, dt: pd.Series, ts: pd.Timestamp) -> pd.Series:
        """Parse and correct datetime values."""
        def _get_baseline(dt: pd.Series) -> int:
            baseline = dt.dt.floor("D").astype(int).unique()
            if baseline.size != 1:
                raise ValueError("conversation spans multiple days")
            return int(baseline[0])

        if pd.api.types.is_numeric_dtype(dt):
            out = pd.to_datetime(dt, origin=ts, unit="s")
            _ = _get_baseline(out)
        else:
            raise NotImplementedError

        out = out.dt.round("L")
        return out

    def sanitize_strings(self, s: pd.Series) -> pd.Series:
        """Sanitize string values."""
        def select_token(s):
            toks = s.split()
            for tok in toks:
                tok = tok.lower()
                if "eksp" in tok:
                    return "expr"
            tok = toks[0] if toks else ""
            if tok.upper() in ("BEGIN", "END"):
                tok = tok.upper()
            else:
                tok = tok.lower()
            return tok

        s = s.copy()
        s[s.isnull()] = ""
        s = s.str.replace(r"[\"':\.-]", r"", regex=True)
        s = s.str.strip()
        s = s.apply(select_token)
        s = s.str.replace(r"o+s+o+b+a+", r"osoba", regex=True)
        s = s.str.replace(r"default\d+", r"default", regex=True)
        s = s.str.replace(r"_+\d+", r"", regex=True)

        # Check tier consistency
        names = s.unique()
        names_under = [ n for n in names if "_" in n ]
        for name in names_under:
            correct, *rest = name.split("_")
            if len(rest) != 1:
                raise ValueError(f"ambiguous tier name '{name}'")
            if correct in names:
                raise ValueError(f"ambiguous tier naming: '{name}' and '{correct}'")
            s[s == name] = correct

        return s

    @classmethod
    def filter(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data frame to leave only relevant
        conversation records.
        """
        df = df[df["tier"].str.startswith("osoba")].copy()
        mask = df["note"] == "BEGIN"
        if mask.any():
            begin_ts = df.loc[mask, "start"].to_numpy()[0]
            df = df[df["start"] >= begin_ts].copy()
        mask = df["note"] == "END"
        if mask.any():
            end_ts = df.loc[mask, "end"].to_numpy()[0]
            df = df[df["end"] <= end_ts].copy()
        return df

    @classmethod
    def merge_contiguous(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Merge contiguous (overlapping) utterances of the same persons."""
        sdf = df.shift(1)
        mask = (df["tier"] == sdf["tier"]).fillna(False) \
            & (df["start"] <= sdf["end"]).fillna(False)
        if mask.any():
            run = (~mask).cumsum()
            spec = {
                "fname":     "first",
                "ts":        "first",
                "condition": "first",
                "tier":      "first",
                "start":     "min",
                "end":       "max",
                "duration":  lambda x: np.nan,
                "note":      "first"
            }
            if (field := "condition") not in df:
                del spec[field]
            df = df.groupby(run).agg(spec).assign(
                duration=lambda df: \
                    (df["end"] - df["start"]).dt.total_seconds()
            )
        return df

    @classmethod
    def add_offsets(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add start/end and floor transfer offsets."""
        sdf = df.shift(1)
        return df.assign(
            offset_start=(df["start"] - sdf["start"]).dt.total_seconds(),
            offset_end=(df["end"] - sdf["end"]).dt.total_seconds(),
            fto=(df["start"] - sdf["end"]).dt.total_seconds()
        )

    @classmethod
    def add_flags(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add flags decribing different types of records."""
        df = df.assign(
            overlap_w=df["offset_end"] <= 0,
            overlap_b=(df["offset_end"] > 0) & (df["fto"] < 0),
        ).pipe(lambda df: df.assign(
            overlap=df["overlap_w"] | df["overlap_b"]
        ))
        df["overlap_time"] = np.nan
        mask = df["overlap_w"]
        df.loc[mask, "overlap_time"] = df.loc[mask, "duration"]
        mask = df["overlap_b"]
        df.loc[mask, "overlap_time"] = np.abs(df.loc[mask, "fto"])

        df["gap_time"] = np.nan
        mask = ~df["overlap"]
        df.loc[mask, "gap_time"] = df.loc[mask, "fto"]

        df["talk_time"] = df["duration"]
        mask = df["overlap_b"]
        df.loc[mask, "talk_time"] -= df.loc[mask, "overlap_time"]
        mask = df["overlap_w"]
        df.loc[mask, "talk_time"] = 0

        sdf = df.shift(1)
        df["event"] = "statement"
        df.loc[df["tier"] == sdf["tier"], "event"] = "continuation"
        df.loc[df["overlap_b"], "event"] = "interruption"
        df.loc[df["overlap_w"], "event"] = "backchannel"
        df["event"] = pd.Categorical(df["event"], categories=[
            "statement",
            "continuation",
            "interruption",
            "backchannel"
        ])
        return df

    @classmethod
    def check_overlaps(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Check if overlapping utterances of the same persons
        are properly merged.
        """
        sdf = df.shift(1)
        mask = (df["tier"] == sdf["tier"]).fillna(False) \
            & (df["start"] <= sdf["end"]).fillna(False)
        assert not mask.any(), \
            "overlaps may occur only between different speakers"
        return df

    @classmethod
    def validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Check if data frame does not contain errors."""
        sdf = df.shift(1)
        # Check if overlaps are only between different speaker
        mask = df["overlap"] & (df["tier"] == sdf["tier"])
        assert not mask.any(), \
            "overlaps may occur only between different speakers"
        # Check if gap time is not defined for interruption/backchanneling
        mask = df["event"].isin(["interruption", "backchanneling"])
        assert df["gap_time"][mask].isnull().all(), \
            "'gap_time' must be undefined for 'interruption'/'backchanneling'"
        # Check if overlap_time is not defined for response/continuation
        mask = df["event"].isin(["response", "continuation"])
        assert df["overlap_time"][mask].isnull().all()
        return df

    @classmethod
    def postprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Postprocess data frame."""
        df = df \
            .pipe(cls.check_overlaps) \
            .sort_values(by=["start", "end"], ascending=True) \
            .reset_index(drop=True) \
            .pipe(cls.check_overlaps) \
            .pipe(cls.add_offsets) \
            .pipe(cls.check_overlaps) \
            .pipe(cls.add_flags) \
            .pipe(cls.validate)
        cats = np.array(df["tier"].unique())
        cats.sort()
        df["tier"] = pd.Categorical(df["tier"], categories=cats)
        return df
