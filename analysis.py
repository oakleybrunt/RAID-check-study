import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu

from config import Config
from csv_parser import parse_data

def mann_whit(dataframe):
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

                    if p < 0.05:
                        print(f"Reject null hypothesis, there is a significant "
                            f"difference in Raw Write Rate between collective "
                            f"buffering and no hints for "
                            f"{node} nodes, {striping}, {raid}")
                        print(p)


def cliffs_delta(x:np.ndarray, y:np.ndarray):
    nx = len(x)
    ny = len(y)
    greater = sum(np.sum(xi > y) for xi in x)
    less = sum(np.sum(xi < y) for xi in x)

    return (greater - less) / (nx * ny)


def permutation_did(no_cb_0, cb_0, no_cb_1, cb_1,
                    n_perm=100, seed=0):

    col = "raw_write_rate_gibs"

    # ==========================================================================
    # 0. Get numpy array of each condition
    # ==========================================================================
    # Create numpy array of each for passing to cliffs_delta calc
    no_cb_0_arr = no_cb_0.select(pl.col(col))[col].to_numpy()
    cb_0_arr = cb_0.select(pl.col(col))[col].to_numpy()
    no_cb_1_arr = no_cb_1.select(pl.col(col))[col].to_numpy()
    cb_1_arr = cb_1.select(pl.col(col))[col].to_numpy()

    # ==========================================================================
    # 1. Compute the observed difference-in-differences
    # ==========================================================================
    # effect of coll buff under condition 0
    d0 = cliffs_delta(no_cb_0_arr, cb_0_arr)

    # effect of coll buff under condition 1
    d1 = cliffs_delta(no_cb_1_arr, cb_1_arr)

    # Difference in Differences
    ## how much coll buff effect changes between conditions
    observed = d1 - d0

    # ==========================================================================
    # 2. prep data for permutation
    # ==========================================================================
    # pool data for each condition
    combined_0 = pl.concat([no_cb_0, cb_0], how="vertical_relaxed")
    combined_1 = pl.concat([no_cb_1, cb_1], how="vertical_relaxed")

    # Set up storage
    perm_stats = []

    # 3. Permutation loop
    for _ in range(n_perm):
        # Randomly reassign the raid labels within each condition, simulating
        # the null hypothesis that collective buffering has no effect
        shuffled_0 = combined_0.with_columns(
            pl.col("hints").shuffle(seed)
        )
        shuffled_1 = combined_1.with_columns(
            pl.col("hints").shuffle(seed)
        )

        # Seperate into psuedo groups for each condition
        x0 = shuffled_0.filter(
            pl.col("hints").str.contains("no hints")
        )[col].to_numpy()
        y0 = shuffled_0.filter(
            pl.col("hints").str.contains("collective buffering")
        )[col].to_numpy()
        x1 = shuffled_1.filter(
            pl.col("hints").str.contains("no hints")
        )[col].to_numpy()
        y1 = shuffled_1.filter(
            pl.col("hints").str.contains("collective buffering")
        )[col].to_numpy()

        # Compute effect of collective buffering under each condition
        perm_d0 = cliffs_delta(x0, y0)
        perm_d1 = cliffs_delta(x1, y1)

        perm_stats.append(perm_d1 - perm_d0)

    # Convert perm_stats to a numpy array for speed
    perm_stats = np.array(perm_stats)

    # ==========================================================================
    # 4. Compute permutation p-value
    # ==========================================================================
    # Two -sided p
    p_value = np.mean(np.abs(perm_stats) >= abs(observed))

    return observed, p_value


if __name__ == '__main__':
    dataframe = parse_data(sample_size=1500, verbose=True)

    for node in Config.get().nodes:
        for striping in Config.get().striping:
            print(f"XIOS NODES : {node}\n"
                  f"STRIPING : {striping}")

            raid_no_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("RAID")) &
                (pl.col("hints").str.contains("no hints")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            raid_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("RAID")) &
                (pl.col("hints").str.contains("collective buffering")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            control_no_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("control")) &
                (pl.col("hints").str.contains("no hints")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )
            control_cb = dataframe.filter(
                ((pl.col("raid_level").str.contains("control")) &
                (pl.col("hints").str.contains("collective buffering")) &
                (pl.col("xios_nodes") == int(node)) &
                (pl.col("striping").str.contains(striping)))
            )

            # Call diff in diffs func
            if int(node) == 1 and striping == 'striped':
                print("NO DATA\n")
                continue
            else:
                observed, p = permutation_did(control_no_cb, control_cb,
                                              raid_no_cb, raid_cb,
                                              n_perm=100)
                print(f"OBSERVED : {observed}")
                print(f"P-VALUE : {p}\n")