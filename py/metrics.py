"""Metrics class."""
# pylint: disable=too-many-public-methods
from typing import Any, Callable, Literal, Iterable
from functools import cached_property
import numpy as np
import pandas as pd
from .dataloader import Dataloader


class Metrics:
    """Metrics class.

    It calculates advanced metrics on conversation
    sequence data frames.

    Attributes
    ----------
    data
        Conversation data frame.
    """
    class Meta:
        """Container for metaprogramming methods."""
        metrics = ()

        @classmethod
        def metric(cls, mode: str, /) -> Callable:
            def decorator(method: Callable) -> Callable:
                cls.metrics = (*cls.metrics, (method.__name__, mode))
                return cached_property(method)
            return decorator

    def __init__(
        self,
        data: pd.DataFrame
    ) -> None:
        """Initialization method."""
        self.data = data

    @cached_property
    def nbdata(self) -> pd.DataFrame:
        """No-backchannel data."""
        df = self.data.copy()
        while df["event"].isin(["backchannel"]).any():
            df = df[df["event"] != "backchannel"].copy() \
                .pipe(Dataloader.merge_contiguous) \
                .pipe(Dataloader.postprocess)
        return df

    @cached_property
    def mdata(self) -> pd.DataFrame:
        """Merged no-backchannel data."""
        df = self.merge_continuations(self.nbdata)
        for col in ("condition", "ts", "fname", "idx"):
            if col in self.data:
                df.insert(0, col, self.data[col].to_numpy()[0])
        return df

    @cached_property
    def sdata(self) -> pd.DataFrame:
        """ABA/ABCD sequence raw data."""
        df = self.mdata.copy()
        # ABA indicators
        df["is_aba"] = pd.Series(pd.factorize(df["tier"])[0]) \
            .rolling(3) \
            .apply(lambda x: x.nunique() == 2) \
            .fillna(0) \
            .astype(int)
        run = df["is_aba"] != df["is_aba"].shift(1).fillna(False)
        df["aba_key"] = np.cumsum(run)
        # ABCD indicators
        df["is_abcd"] = pd.Series(pd.factorize(df["tier"])[0]) \
            .rolling(4) \
            .apply(lambda x: x.nunique() == 4) \
            .fillna(0) \
            .astype(int)
        run = df["is_abcd"] != df["is_abcd"].shift(1).fillna(False)
        df["abcd_key"] = np.cumsum(run)
        # M(4,3) indicators
        df["is_m43"] = pd.Series(pd.factorize(df["tier"])[0]) \
            .rolling(4) \
            .apply(lambda x: x.nunique() == 3) \
            .fillna(0) \
            .astype(int)
        run = df["is_m43"] != df["is_m43"].shift(1).fillna(False)
        df["m43_key"] = np.cumsum(run)
        # Other measures
        duration = (df["end"] - df["start"]).dt.total_seconds()
        df.insert(6, "duration", duration)
        df.insert(8, "overlap", df["duration"] - df["talk_time"])
        return df

    @cached_property
    def odata2(self) -> pd.DataFrame:
        """Overlap data."""
        df = self.get_overlaps(self.data, k=2)
        df["i"] = df["tier_seq"].map(lambda x: x[0])
        df["j"] = df["tier_seq"].map(lambda x: x[1])
        return df

    @cached_property
    def aba(self) -> pd.DataFrame:
        """ABA sequence data."""
        df  = self.sdata
        gdf = df[df["is_aba"] == 1] \
            .groupby(level=0) \
            .apply(lambda d: df[d.index.min()-2:d.index.max()+1]) \
            .groupby(level=0) \
            .agg({
                "start": np.min,
                "end":   np.max,
                "event": tuple
            })
        duration = (gdf["end"] - gdf["start"]).dt.total_seconds()
        gdf.insert(gdf.shape[1]-1, "duration", duration)
        return gdf

    @cached_property
    def abcd(self) -> pd.DataFrame:
        """ABCD data."""
        df  = self.sdata
        gdf = df[df["is_abcd"] == 1] \
            .groupby(level=0) \
            .apply(lambda d: df[d.index.min()-3:d.index.max()+1]) \
            .groupby(level=0) \
            .agg({
                "start": np.min,
                "end":   np.max,
                "event": tuple
            })
        duration = (gdf["end"] - gdf["start"]).dt.total_seconds()
        gdf.insert(gdf.shape[1]-1, "duration", duration)
        return gdf

    @cached_property
    def user_ftime_ranking(self) -> pd.Series:
        time = self.nbdata.groupby("tier")["duration"] \
            .sum() \
            .sort_values(ascending=False)
        return time / time.sum()

    @cached_property
    def user_unit_ranking(self) -> pd.Series:
        units = self.nbdata.groupby("tier") \
            .size() \
            .sort_values(ascending=False)
        return units / units.sum()

    @property
    def metrics(self) -> tuple[tuple[str, str], ...]:
        return self.Meta.metrics

    @property
    def global_metrics(self) -> tuple[str, ...]:
        return tuple(name for name, mode in self.metrics if mode == "global")
    @property
    def user_metrics(self) -> tuple[str, ...]:
        return tuple(name for name, mode in self.metrics if mode == "user")
    @property
    def pairwise_metrics(self) -> tuple[str, ...]:
        return tuple(name for name, mode in self.metrics if mode == "pairwise")

    def dump(
        self,
        mode: Literal["gloabl", "user", "pairwise"] = "global"
    ) -> dict[str, Any]:
        """Dump to a dictionary."""
        # pylint: disable=too-many-branches
        if mode == "global":
            metrics = self.global_metrics
        elif mode == "user":
            metrics = self.user_metrics
        elif mode == "pairwise":
            metrics = self.pairwise_metrics
        else:
            raise ValueError(f"unknown value '{mode}' of 'mode'")
        out = pd.DataFrame({
            f: self.data[f] for f in ("idx", "fname", "ts", "condition", "tier")
            if f in self.data
        }).reset_index(drop=True)
        if mode == "global":
            out = out.drop(columns="tier").iloc[0]
        elif mode == "user":
            out = out.groupby("tier").first()
        else:
            sdata = self.sdata
            out = self.odata2[["i", "j"]].assign(**{
                col: sdata[col].to_numpy()[0]
                for col in ("idx", "fname", "ts", "condition")
                if col in sdata
            }).groupby(["i", "j"]).first()
        for metric in metrics:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = getattr(self, metric)
                if "_lr_" in metric:
                    new_metric = metric.replace("_lr_", "_sqrt_")
                    new_x = np.sqrt(np.exp(x))
                    out[new_metric] = new_x
                if np.isscalar(x):
                    x = np.nan if np.isinf(x) else x
                else:
                    x[np.isinf(x)] = np.nan
                out[metric] = x
            if metric.startswith("p_n_") or metric.startswith("u_n_"):
                out[metric] = out[metric]\
                    .fillna(0, inplace=False)\
                    .astype(int)

        if mode == "user":
            out.reset_index(drop=False, inplace=True)
            out.insert(3, "tier", out.pop("tier"))
        elif mode == "pairwise":
            out.reset_index(drop=False, inplace=True)
            out.insert(4, "j", out.pop("j"))
            out.insert(3, "i", out.pop("i"))

        return out

    # Global metrics ----------------------------------------------------------

    @Meta.metric("global")
    def time_total(self) -> float:
        return (self.data["end"].max() - self.data["start"].min()) \
            .total_seconds()

    @Meta.metric("global")
    def time_talk(self) -> float:
        return self.nbdata["talk_time"].sum()
    @Meta.metric("global")
    def ftime_talk(self) -> float:
        return self.time_talk / self.time_total

    @Meta.metric("global")
    def time_silence(self) -> float:
        return self.time_total - self.time_talk
    @Meta.metric("global")
    def ftime_silence(self) -> float:
        return self.time_silence / self.time_total

    @Meta.metric("global")
    def time_overlap4(self) -> float:
        df = self.get_overlaps(self.data, 4)
        return df["duration"].sum()
    @Meta.metric("global")
    def ftime_overlap4(self) -> float:
        return self.time_overlap4 / self.time_talk

    @Meta.metric("global")
    def time_overlap3(self) -> float:
        df = self.get_overlaps(self.data, 3)
        return df["duration"].sum() - 3*self.time_overlap4
    @Meta.metric("global")
    def ftime_overlap3(self) -> float:
        return self.time_overlap3 / self.time_talk

    @Meta.metric("global")
    def time_overlap2(self) -> float:
        df = self.get_overlaps(self.data, 2)
        return df["duration"].sum() \
            - 2*self.time_overlap3 \
            - 5*self.time_overlap4
    @Meta.metric("global")
    def ftime_overlap2(self) -> float:
        return self.time_overlap2 / self.time_talk

    @Meta.metric("global")
    def time_overlap(self) -> float:
        return self.time_overlap2 + self.time_overlap3 + self.time_overlap4
    @Meta.metric("global")
    def ftime_overlap(self) -> float:
        return self.time_overlap / self.time_talk

    @Meta.metric("global")
    def time_average(self) -> float:
        return self.nbdata["duration"].mean()

    @Meta.metric("global")
    def n_units(self) -> int:
        return len(self.data)

    @Meta.metric("global")
    def n_units_nb(self) -> int:
        return len(self.nbdata)

    @Meta.metric("global")
    def n_statements(self) -> int:
        return (self.data["event"] == "statement").sum()
    @Meta.metric("global")
    def f_statements(self) -> int:
        return self.n_statements / self.n_units

    @Meta.metric("global")
    def n_continuations(self) -> int:
        return (self.data["event"] == "continuation").sum()
    @Meta.metric("global")
    def f_continuations(self) -> int:
        return self.n_continuations / self.n_units

    @Meta.metric("global")
    def n_interruptions(self) -> int:
        return (self.data["event"] == "interruption").sum()
    @Meta.metric("global")
    def f_interruptions(self) -> int:
        return self.n_interruptions / self.n_units

    @Meta.metric("global")
    def n_backchannels(self) -> int:
        return (self.data["event"] == "backchannel").sum()
    @Meta.metric("global")
    def f_backchannels(self) -> int:
        return self.n_backchannels / self.n_units

    @Meta.metric("global")
    def tt_equality(self) -> float:
        return self.entropy(self.nbdata.groupby("tier")["duration"].sum()) / 2
    @Meta.metric("global")
    def tt_equality_units(self) -> float:
        return self.entropy(self.nbdata["tier"].value_counts()) / 2

    @Meta.metric("global")
    def tt_freedom(self) -> float:
        df = self.nbdata
        x = df["tier"]
        return 1 - self.info_gain(x.shift(1), x)

    @Meta.metric("global")
    def tt_freedom_statement(self) -> float:
        df = self.nbdata
        mask = df["event"].isin(["statement"])
        x0 = df.shift(1).loc[mask, "tier"]
        x1 = df.loc[mask, "tier"]
        return 1 - self.info_gain(x0, x1)

    @Meta.metric("global")
    def tt_freedom_interruption(self) -> float:
        df = self.nbdata
        mask = df["event"] == "interruption"
        x0 = df.shift(1).loc[mask, "tier"]
        x1 = df.loc[mask, "tier"]
        return 1 - self.info_gain(x0, x1)

    @Meta.metric("global")
    def tt_freedom_backchannel(self) -> float:
        mask = self.data["event"] == "backchannel"
        x0 = self.data.shift(1).loc[mask, "tier"]
        x1 = self.data.loc[mask, "tier"]
        return 1 - self.info_gain(x0, x1)

    @Meta.metric("global")
    def uftime1(self) -> float:
        return self.user_ftime_ranking.iloc[0]
    @Meta.metric("global")
    def uftime2(self) -> float:
        return self.user_ftime_ranking.iloc[1]
    @Meta.metric("global")
    def uftime3(self) -> float:
        return self.user_ftime_ranking.iloc[2]
    @Meta.metric("global")
    def uftime4(self) -> float:
        return self.user_ftime_ranking.iloc[3]

    @Meta.metric("global")
    def ufunits1(self) -> float:
        return self.user_unit_ranking.iloc[0]
    @Meta.metric("global")
    def ufunits2(self) -> float:
        return self.user_unit_ranking.iloc[1]
    @Meta.metric("global")
    def ufunits3(self) -> float:
        return self.user_unit_ranking.iloc[2]
    @Meta.metric("global")
    def ufunits4(self) -> float:
        return self.user_unit_ranking.iloc[3]

    @Meta.metric("global")
    def f_aba(self) -> float:
        df = self.sdata
        return df["is_aba"].sum() / (len(df)-2)

    @Meta.metric("global")
    def aba_mean_excess(self) -> float:
        return self.sdata \
            .query("is_aba == 1") \
            .groupby("aba_key") \
            .size() \
            .sub(1) \
            .mean()

    @cached_property
    def _aba_mean_time_vec(self) -> pd.Series:
        return self.sdata.groupby("is_aba")["talk_time"] \
            .apply(lambda s: s.rolling(3).sum().mean())
    @Meta.metric("global")
    def aba_mean_time(self) -> float:
        return self._aba_mean_time_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonaba_mean_time(self) -> float:
        return self._aba_mean_time_vec.get(0, np.nan)
    @Meta.metric("global")
    def aba_mean_time_lr(self) -> float:
        time = self._aba_mean_time_vec
        if any(pd.isnull(time)):
            return 0
        return np.log(time[1] / time[0])

    @cached_property
    def _aba_entropy_vec(self) -> pd.Series:
        return self.sdata.groupby("is_aba")["duration"].apply(lambda s: s \
            .rolling(3) \
            .apply(lambda x: self.entropy(x, normalized=True)) \
            .mean()
        )
    @Meta.metric("global")
    def aba_entropy(self) -> float:
        return self._aba_entropy_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonaba_entropy(self) -> float:
        return self._aba_entropy_vec.get(0, np.nan)
    @Meta.metric("global")
    def aba_entropy_lr(self) -> float:
        ent = self._aba_entropy_vec
        return np.log(ent[1] / ent[0])

    @cached_property
    def _aba_f_interruption_vec(self) -> pd.Series:
        df = self.sdata
        rate = df.assign(
            interruption=lambda df:
            (df["event"] == "interruption").rolling(3).mean()
        ).groupby("is_aba")["interruption"].mean()
        eps = 1 / (len(df)+1)
        return rate.clip(eps, 1-eps)
    @Meta.metric("global")
    def aba_f_interruption_lr(self) -> float:
        return self._aba_f_interruption_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonaba_f_interruption(self) -> float:
        return self._aba_f_interruption_vec.get(0, np.nan)

    @Meta.metric("global")
    def f_abcd(self) -> float:
        df = self.sdata
        return df["is_abcd"].sum() / (len(df)-3)

    @Meta.metric("global")
    def abcd_mean_excess(self) -> float:
        return self.sdata \
            .query("is_abcd == 1") \
            .groupby("abcd_key") \
            .size() \
            .sub(1) \
            .mean()

    @cached_property
    def _abcd_mean_time_vec(self) -> pd.Series:
        return self.sdata.groupby("is_abcd")["talk_time"] \
            .apply(lambda s: s.rolling(4).sum().mean())
    @Meta.metric("global")
    def abcd_mean_time(self) -> float:
        return self._abcd_mean_time_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonabcd_mean_time(self) -> float:
        return self._abcd_mean_time_vec.get(0, np.nan)
    @Meta.metric("global")
    def abcd_mean_time_lr(self) -> float:
        time = self._abcd_mean_time_vec
        if any(pd.isnull(time)):
            return 0
        return np.log(time.get(1, np.nan) / time.get(0, np.nan))

    @cached_property
    def _abcd_entropy_vec(self) -> pd.Series:
        return self.sdata.groupby("is_abcd")["duration"].apply(lambda s: s \
            .rolling(4) \
            .apply(lambda x: self.entropy(x, normalized=True)) \
            .mean()
        )
    @Meta.metric("global")
    def abcd_entropy(self) -> float:
        return self._abcd_entropy_vec.get(1, np.nan)
    @Meta.metric("gloabl")
    def nonabcd_entropy(self) -> float:
        return self._abcd_entropy_vec.get(0, np.nan)
    @Meta.metric("global")
    def abcd_entropy_lr(self) -> float:
        ent = self._abcd_entropy_vec
        return np.log(ent.get(1, np.nan) / ent.get(0, np.nan))

    @cached_property
    def _abcd_f_interruption_vec(self) -> pd.Series:
        df = self.sdata
        rate = df.assign(
            interruption=lambda df:
            (df["event"] == "interruption").rolling(4).mean()
        ).groupby("is_abcd")["interruption"].mean()
        eps = 1 / (len(df)+1)
        return rate.clip(eps, 1-eps)
    @Meta.metric("global")
    def abcd_f_interruption(self) -> float:
        return self._abcd_f_interruption_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonabcd_f_interruption(self) -> float:
        return self._abcd_f_interruption_vec.get(0, np.nan)
    @Meta.metric("global")
    def abcd_f_interruption_lr(self) -> float:
        rate = self._abcd_f_interruption_vec
        return np.log(rate.get(1, np.nan) / rate.get(0, np.nan))

    @Meta.metric("global")
    def f_m43(self) -> float:
        df = self.sdata
        return df["is_m43"].sum() / (len(df)-3)

    @Meta.metric("global")
    def m43_mean_excess(self) -> float:
        return self.sdata \
            .query("is_m43 == 1") \
            .groupby("m43_key") \
            .size() \
            .sub(1) \
            .mean()

    @cached_property
    def _m43_mean_time_vec(self) -> pd.Series:
        return self.sdata.groupby("is_m43")["talk_time"] \
            .apply(lambda s: s.rolling(4).sum().mean())
    @Meta.metric("global")
    def m43_mean_time(self) -> float:
        return self._m43_mean_time_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonm43_mean_time(self) -> float:
        return self._m43_mean_time_vec.get(0, np.nan)
    @Meta.metric("global")
    def m43_mean_time_lr(self) -> float:
        time = self._m43_mean_time_vec
        if any(pd.isnull(time)):
            return 0
        return np.log(time.get(1, np.nan) / time.get(0, np.nan))

    @cached_property
    def _m43_entropy_vec(self) -> pd.Series:
        return self.sdata.groupby("is_m43")["duration"].apply(lambda s: s \
            .rolling(4) \
            .apply(lambda x: self.entropy(x, normalized=True)) \
            .mean()
        )
    @Meta.metric("global")
    def m43_entropy(self) -> float:
        return self._m43_entropy_vec.get(1, np.nan)
    @Meta.metric("gloabl")
    def nonm43_entropy(self) -> float:
        return self._m43_entropy_vec.get(0, np.nan)
    @Meta.metric("global")
    def m43_entropy_lr(self) -> float:
        ent = self._m43_entropy_vec
        return np.log(ent.get(1, np.nan) / ent.get(0, np.nan))

    @cached_property
    def _m43_f_interruption_vec(self) -> pd.Series:
        df = self.sdata
        rate = df.assign(
            interruption=lambda df:
            (df["event"] == "interruption").rolling(4).mean()
        ).groupby("is_m43")["interruption"].mean()
        eps = 1 / (len(df)+1)
        return rate.clip(eps, 1-eps)
    @Meta.metric("global")
    def m43_f_interruption(self) -> float:
        return self._m43_f_interruption_vec.get(1, np.nan)
    @Meta.metric("global")
    def nonm43_f_interruption(self) -> float:
        return self._m43_f_interruption_vec.get(0, np.nan)
    @Meta.metric("global")
    def m43_f_interruption_lr(self) -> float:
        rate = self._m43_f_interruption_vec
        return np.log(rate.get(1, np.nan) / rate.get(0, np.nan))

    # User metrics ------------------------------------------------------------

    @Meta.metric("user")
    def u_time_talk(self) -> pd.Series:
        return self.data.groupby("tier")["talk_time"].sum()
    @Meta.metric("user")
    def u_ftime_talk(self) -> pd.Series:
        return self.u_time_talk / self.data["talk_time"].sum()

    @Meta.metric("user")
    def u_n_statements(self) -> pd.Series:
        return self.data["tier"].value_counts()
    @Meta.metric("user")
    def u_f_statements(self) -> pd.Series:
        return self.u_n_statements / len(self.data)

    @Meta.metric("user")
    def u_mean_time(self) -> pd.Series:
        df = self.data[self.data["event"] != "backchannel"]
        return df.groupby("tier")["talk_time"].mean()

    @Meta.metric("user")
    def u_n_pauses(self) -> pd.Series:
        df = self.data[self.data["event"] == "continuation"]
        return df["tier"].value_counts()

    @Meta.metric("user")
    def u_pause_mean_time(self) -> pd.Series:
        df = self.data[self.data["event"] == "continuation"]
        return df.groupby("tier")["gap_time"].mean()

    @Meta.metric("user")
    def u_n_backchannel(self) -> pd.Series:
        df = self.data[self.data["event"] == "backchannel"]
        df = df[df["overlap_time"] < 2]
        return df["tier"].value_counts()
    @Meta.metric("user")
    def u_lr_backchannel(self) -> pd.Series:
        x = (self.u_n_backchannel / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_backchanneled(self) -> pd.Series:
        df = self.odata2
        df = df[df["duration"] < 2]
        df = df[df["event"] == "backchannel"]
        return df["i"].value_counts()
    @Meta.metric("user")
    def u_lr_backchanneled(self) -> pd.Series:
        x = (self.u_n_backchanneled / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_interrupt_success(self) -> pd.Series:
        df = self.data[self.data["event"] == "interruption"]
        return df["tier"].value_counts()
    @Meta.metric("user")
    def u_lr_interrupt_success(self) -> pd.Series:
        x = (self.u_n_interrupt_success / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_interrupt_fail(self) -> pd.Series:
        df = self.data[self.data["event"] == "backchannel"]
        df = df[df["overlap_time"] >= 2]
        return df["tier"].value_counts()
    @Meta.metric("user")
    def u_lr_interrupt_fail(self) -> pd.Series:
        x = (self.u_n_interrupt_fail / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_interrupted_success(self) -> pd.Series:
        df = self.odata2
        df = df[df["event"] == "interruption"]
        return df["i"].value_counts()
    @Meta.metric("user")
    def u_lr_interrupted_success(self) -> pd.Series:
        x = (self.u_n_interrupted_success / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_interrupted_fail(self) -> pd.Series:
        df = self.odata2
        df = df[df["event"] == "backchannel"]
        df = df[df["duration"] >= 2]
        return df["i"].value_counts()
    @Meta.metric("user")
    def u_lr_interrupted_fail(self) -> pd.Series:
        x = (self.u_n_interrupted_fail / self.u_n_statements).fillna(0)
        return np.log(x)

    @Meta.metric("user")
    def u_n_aba(self) -> pd.Series:
        df = self.sdata
        df = df[df["is_aba"] == 1]
        return df["tier"].value_counts()
    @Meta.metric("user")
    def u_f_aba(self) -> pd.Series:
        return self.u_n_aba / self.u_n_statements

    @Meta.metric("user")
    def u_n_abcd(self) -> pd.Series:
        df = self.sdata
        df = df[df["is_abcd"] == 1]
        return df["tier"].value_counts()
    @Meta.metric("user")
    def u_f_abcd(self) -> pd.Series:
        return self.u_n_abcd / self.u_n_statements

    # Pairwise metrics --------------------------------------------------------

    @Meta.metric("pairwise")
    def p_n_backchannel(self) -> pd.Series:
        df = self.odata2
        df = df[df["event"] == "backchannel"]
        df = df[df["duration"] < 2]
        return df.groupby(["i", "j"]).size()
    @Meta.metric("pairwise")
    def p_lr_backchannel(self) -> pd.Series:
        # pylint: disable=no-member
        x = self.p_n_backchannel
        n = self.u_n_statements
        i = x.index.get_level_values("i")
        return x / n[i].to_numpy()

    @Meta.metric("pairwise")
    def p_n_interrupt_success(self) -> pd.Series:
        df = self.odata2
        df = df[df["event"] == "interruption"]
        return df.groupby(["i", "j"]).size()
    @Meta.metric("pairwise")
    def p_lr_success(self) -> pd.Series:
        # pylint: disable=no-member
        x = self.p_n_interrupt_success
        n = self.u_n_statements
        i = x.index.get_level_values("i")
        return x / n[i].to_numpy()

    @Meta.metric("pairwise")
    def p_n_interrupt_fail(self) -> pd.Series:
        df = self.odata2
        df = df[df["event"] == "backchannel"]
        df = df[df["duration"] <= 2]
        return df.groupby(["i", "j"]).size()
    @Meta.metric("pairwise")
    def p_lr_interrupt_fail(self) -> pd.Series:
        # pylint: disable=no-member
        x = self.p_n_interrupt_fail
        n = self.u_n_statements
        i = x.index.get_level_values("i")
        return x / n[i].to_numpy()


    # Auxiliary methods -------------------------------------------------------

    @staticmethod
    def entropy(p: np.ndarray, *, normalized: bool = False) -> float:
        """Entropy of a probability distribution.

        Parameters
        ----------
        p
            Array-like sequence of non-negative floating
            point values summing up to ``1``.
        """
        p = np.array(p)
        if (p < 0).any():
            raise ValueError("'p' has to be non-negative")
        p = p / p.sum()
        if not np.isclose(p.sum(), 1):
            raise ValueError("'p' has to sum up to 1")

        p = np.clip(p, 0, 1)
        p[p == 0] = 1
        h = -np.sum(p*np.log2(p))
        if normalized:
            h /= np.log2(len(p))
        return h

    @classmethod
    def conditional_entropy(
        cls,
        x0: np.ndarray,
        x1: np.ndarray,
        **kwds: Any
    ) -> float:
        """Conditional entropy of transitions."""
        p = x0.value_counts()
        p /= p.sum()
        C = pd.crosstab(x0, x1)
        P = C.div(C.sum(axis=1), axis=0)
        h = P.apply(cls.entropy, **kwds, axis=1)
        return (p*h).sum()

    @classmethod
    def info_gain(cls, x0: np.ndarray, x1: np.ndarray) -> float:
        """Information gain."""
        if not x0.size or not x1.size:
            return np.nan
        c  = x1.value_counts()
        h  = cls.entropy(c / c.sum())
        ch = cls.conditional_entropy(x0, x1)
        return (h - ch) / h

    @classmethod
    def turn_taking_equality(cls, df: pd.DataFrame) -> float:
        """Calculate turn taking equality from
        a data frame with tiers and talk times.
        """
        time = df.groupby("tier")["talk_time"].sum()
        time = time / time.sum()
        dev  = ((time - .25)**2).sum()
        mdev = .75**2 + 3*.25**2
        return 1 - dev / mdev

    @staticmethod
    def find_overlaps(df: pd.DataFrame) -> pd.DataFrame:
        """Find overlaps in a data frame

        This method can be applied multiple times
        to find overlaps involving more than two speakers.
        """
        def tolist(x):
            if not isinstance(x, str) and isinstance(x, Iterable):
                return list(x)
            if pd.isnull(x) or not x:
                return []
            return [x]

        df0 = df[["tier", "start", "end", "event"]].copy()
        df0["tier"] = df0["tier"].to_numpy()
        df1 = df0.shift(1)

        # pylint: disable=redefined-argument-from-local
        for df in (df0, df1):
            if "tier_seq" not in df:
                df["tier_seq"] = df["tier"]
            df["tier_seq"] = df["tier_seq"].map(tolist)
            df["tier"] = df["tier_seq"].map(set)

        mask = df0["start"] < df1["end"]
        df0  = df0[mask]
        df1  = df1[mask]

        out = df0.assign(tier_seq=df1["tier_seq"] + df0["tier_seq"]) \
            .assign(
                tier=lambda df: df["tier_seq"].map(set),
                end=pd.concat([df0["end"], df1["end"]], axis=1).min(axis=1)
            )
        out["duration"] = (out["end"] - out["start"]).dt.total_seconds()
        return out

    @classmethod
    def get_overlaps(cls, df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
        """Get ``k`` speakers overlaps data frame."""
        odf = df.copy()
        for i in range(2, k+1):
            odf = cls.find_overlaps(odf)
            odf = odf[odf["tier"].map(len) == i].copy()
        return odf

    @classmethod
    def merge_continuations(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Merge adjacent continuation events
        into single statements.
        """
        df = df[["tier", "start", "end", "talk_time", "event"]].copy()
        run = np.cumsum(df["event"] != "continuation")
        return df.groupby(run).apply(lambda d: pd.Series({
            "tier":     d["tier"].to_numpy()[0],
            "start":     d["start"].min(),
            "end":       d["end"].max(),
            "talk_time": d["talk_time"].sum(),
            "event":     d["event"].to_numpy()[0]
        })).reset_index(drop=True)
