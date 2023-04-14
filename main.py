"""Main script."""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from py import __path__
from py.dataloader import Dataloader
from py.validator import Validator
from py.metrics import Metrics
from py.uniseq import Uniseq

TYPE = "structured"
HERE = Path(__path__[0]).parent
PROD = HERE/"production"/TYPE
DATA = PROD/"data"

dloader = Dataloader(DATA, PROD/"propmap-groups.csv")

df = next(iter(dloader))
Metrics(df).dump(mode="pairwise")
