"""Make group-level data."""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from py import __path__
from py.dataloader import Dataloader
from py.metrics import Metrics
from py.uniseq import Uniseq

HERE = Path(__file__).parent
DATA = HERE/"data"
RAW  = DATA/"raw"

config = {
    "header": None,
    "sep": ",",
    "usecols": (0, 3, 5, 8)
}
dloader = Dataloader(RAW, **config)
df = list(dloader)[0]

D = pd.concat(list(dloader), axis=0)
M = pd.DataFrame(
    Metrics(df).dump()
    for df in tqdm(dloader, total=dloader.n_paths)
)
S = pd.concat([
    Metrics(df).sdata
    for df in tqdm(dloader, total=dloader.n_paths)
])
U = pd.concat([
    Uniseq(df).dump()
    for df in tqdm(dloader, total=dloader.n_paths)
], axis=0)

D.to_csv(DATA/"groups-raw.tsv", sep="\t", index=False)
M.to_csv(DATA/"groups.tsv", sep="\t", index=False)
S.to_csv(DATA/"sequences.tsv", sep="\t", index=False)
U.to_csv(DATA/"uniseq.tsv", sep="\t", index=False)
