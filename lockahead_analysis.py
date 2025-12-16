import polars as pl
import numpy as np
from config import Config
from csv_parser import parse_data
from analysis import cliffs_delta


def permutation_single(x, y, n_perm=100, seed=0):

    col = "raw_write_rate_gibs"

    # ==========================================================================
    # 0. Get numpy array of each condition
    # ==========================================================================
    # Create numpy array of each for passing to cliffs_delta calc
    x_arr = x.select(pl.col(col))[col].to_numpy()
    y_arr = y.select(pl.col(col))[col].to_numpy()

    # ==========================================================================
    # 1 Get effect of RAID on write rate for lockahead and standard
    # ==========================================================================
    # effect of RAID on write rate (standard locking)
    observed = cliffs_delta(y_arr, x_arr)

    # ==========================================================================
    # 2. prep data for permutation
    # ==========================================================================
    combined = pl.concat([x, y], how="vertical_relaxed")

    # Set up storage
    perm_stats = []

    # 3. Permutation loop
    for i in range(n_perm):
        # Randomly reassign the locking labels within each condition, simulating
        # the null hypothesis that Lockahead has no effect
        shuffled = combined.with_columns(
            pl.col("hints").shuffle(seed)
        )

        # Increase the seed by 1 otherwise we get the same shuffling again
        seed += 1

        # Seperate into psuedo groups for each condition
        x0 = shuffled.filter(
            pl.col("hints").str.contains("Standard Locking")
        )[col].to_numpy()
        y0 = shuffled.filter(
            pl.col("hints").str.contains("Lockahead")
        )[col].to_numpy()

        perm_stats.append(cliffs_delta(x0, y0))

    # Convert perm_stats to a numpy array for speed
    perm_stats = np.array(perm_stats)

    # ==========================================================================
    # 4. Compute permutation p-value
    # ==========================================================================
    # Two -sided p
    p_value = np.mean(np.abs(perm_stats) >= abs(observed))

    return observed, p_value


if __name__ == '__main__':
    dataframe = parse_data(sample_size=3500, verbose=False)
    single_perm_results = pl.DataFrame()
    did_results_frame = pl.DataFrame()

    for node in Config.get().nodes:
        for striping in Config.get().striping:

            raid_no_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("RAID")) &
                (pl.col("hints").str.contains("Standard Locking")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            raid_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("RAID")) &
                (pl.col("hints").str.contains("Lockahead")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            control_no_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("control")) &
                (pl.col("hints").str.contains("Standard Locking")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            control_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("control")) &
                (pl.col("hints").str.contains("Lockahead")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )

            # Call diff in diffs func
            if int(node) == 1 and striping == 'striped':
                continue
            else:
                # Do single permutation tests
                observed, p = permutation_single(control_no_cb, control_cb,
                                                n_perm=1000)
                one_result_std = pl.DataFrame([
                    pl.Series("observed", [observed], pl.Float64),
                    pl.Series("p", [p], pl.Float64),
                    pl.Series("striping", [striping], pl.String),
                    pl.Series("xios_nodes", [int(node)], pl.Int32),
                    pl.Series("sample_size", [3500], pl.Int32),
                    pl.Series("raid_level", ["control"], pl.String)
                ])
                single_perm_results = single_perm_results.vstack(one_result_std)

                observed, p = permutation_single(raid_no_cb, raid_cb,
                                                n_perm=1000)
                one_result_lock = pl.DataFrame([
                    pl.Series("observed", [observed], pl.Float64),
                    pl.Series("p", [p], pl.Float64),
                    pl.Series("striping", [striping], pl.String),
                    pl.Series("xios_nodes", [int(node)], pl.Int32),
                    pl.Series("sample_size", [3500], pl.Int32),
                    pl.Series("raid_level", ["RAID"], pl.String)
                ])
                single_perm_results = single_perm_results.vstack(one_result_lock)


    with pl.Config(tbl_rows=-1):
        print(single_perm_results)


    single_perm_results.write_csv("locking_effect_sizes.csv")