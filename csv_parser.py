from pathlib import Path
import polars as pl

from config import Config


def parse_data(file=None, sample_size=None, verbose=False):

    dataframe = pl.DataFrame()

    for node in Config.get().nodes:
        for raid in Config.get().raid_check:
            for stripe in Config.get().striping:
                for filename in Config.get().filenames:

                    if filename == 'coll_buff':
                        hints = 'Lockahead'
                    else:
                        hints = 'Standard Locking'

                    csv_path = Path(f"data/{node}_node_xios/{raid}/{stripe}/{filename}.csv")

                    if csv_path.exists():
                        if verbose:
                            print(f"Parsing '{csv_path}'")
                        csv_data = pl.read_csv(
                            csv_path,
                            infer_schema_length=100000)

                        # Clean up the data

                        ## Convert write_rate to floating point (some strings may be
                        ## present) and date to datetime
                        csv_data = csv_data.with_columns(
                            pl.col("raw_write_rate_mibs").cast(
                                pl.Float64, strict=False).alias(
                                    "raw_write_rate_mibs"),
                            pl.col("created").str.to_datetime(
                                format="%Y-%m-%d %H:%M:%S%.f"),
                        )

                        csv_data = csv_data.drop_nulls()

                        # Convert write rate to GiBs by dividing by 1024
                        csv_data = csv_data.with_columns(
                            raw_write_rate_gibs = pl.col(
                                "raw_write_rate_mibs").truediv(1024)
                        )

                        # check for normal distribution in each file
                        # names = csv_data.unique("filename")['filename'].to_list()
                        # _, p = normaltest(pl.Series(csv_data.select("raw_write_rate_gibs")).to_list())

                        # if p <= 0.00174:
                        #     print(f"NON-NORMAL Distribution")
                        # else:
                        #     print(f"NORMAL Distributions")

                        if raid == 'RAID':
                            csv_data = csv_data.filter(
                                (pl.col("created").is_between(Config.get().RAID_START, Config.get().RAID_END))
                                | (pl.col("created").ge(Config.get().RAID_START_2)),
                            )
                        else:
                            csv_data = csv_data.filter(
                                ~((pl.col("created").is_between(Config.get().RAID_START, Config.get().RAID_END)) |
                                (pl.col("created").is_between(Config.get().RAID_START_2, Config.get().RAID_END_2)))
                            )

                        # Filter for specified file (if applicable)
                        if file is not None:
                            csv_data = csv_data.filter(
                                pl.col("filename").str.contains("lfric_gl_std_levs_diags_1hr")
                            )

                        # Get an equal sample size
                        if sample_size:
                            data_len = sample_size
                            csv_data = csv_data.sample(n=data_len, seed=72)
                        else:
                            data_len = len(csv_data)

                        # Calculate mean and std deviatipn
                        # mean = csv_data.select(pl.mean('raw_write_rate_gibs'))['raw_write_rate_gibs'].to_list()[0]
                        # std_dev = csv_data.select(pl.std('raw_write_rate_gibs'))['raw_write_rate_gibs'].to_list()[0]

                        stats_frame = pl.DataFrame([
                            pl.Series(csv_data.select("raw_write_rate_gibs")),
                        ])

                        # Print some descriptive statistics
                        if verbose:
                            with pl.Config(tbl_rows=-1):
                                print(stats_frame.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

                        stats_frame = pl.DataFrame([
                            pl.Series(csv_data.select("raw_write_rate_gibs")),
                            pl.Series(csv_data.select(pl.col("filename"))),
                            pl.Series("raid_level", [raid] * data_len, pl.String),
                            pl.Series("xios_nodes", [int(node)] * data_len, pl.Int32),
                            pl.Series("striping", [stripe] * data_len, pl.String),
                            pl.Series("hints", [hints] * data_len, pl.String),
                        ])

                        # Add to global dataframe
                        dataframe = dataframe.vstack(stats_frame)

                    else:
                        continue

    return dataframe


__all__ = [
    'parse_data'
]
