"""Make user-level data."""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from py import __path__
from py.dataloader import Dataloader
from py.metrics import Metrics


HERE = Path(__file__).parent
DATA = HERE/"data"
RAW  = DATA/"raw"

config = {
    "header": None,
    "usecols": (0, 3, 5, 8),
    "sep": ","
}
dloader = Dataloader(RAW, **config)

U = pd.concat([
    Metrics(df).dump(mode="user")
    for df in tqdm(dloader)
], ignore_index=True)

U.to_csv(DATA/"users.tsv", sep="\t", index=False)
