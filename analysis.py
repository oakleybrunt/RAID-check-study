import polars as pl
import pandas as pd
from scipy.stats import mannwhitneyu

from config import Config
from csv_parser import parse_data

dataframe = parse_data()

for node in Config.get().nodes:
    for striping in Config.get().striping:
        for raid in Config.get().raid_check:

            cb = dataframe.filter(
                ((pl.col("hints").str.contains("collective buffering")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)) &
                (pl.col("raid_level").str.contains(raid)))
            ).to_pandas()

            no_cb = dataframe.filter(
                ((pl.col("hints").str.contains("no hints")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)) &
                (pl.col("raid_level").str.contains(raid)))
            ).to_pandas()

            if len(cb) < 1 or len(no_cb) < 1:
                continue
            else:
                u, p = mannwhitneyu(no_cb['raw_write_rate_gibs'],
                                    cb['raw_write_rate_gibs'],
                                    alternative="two-sided")
                print(p)