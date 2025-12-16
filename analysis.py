import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

from config import Config
from csv_parser import parse_data


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
        # the null hypothesis that Lockahead has no effect
        shuffled_0 = combined_0.with_columns(
            pl.col("hints").shuffle(seed)
        )
        shuffled_1 = combined_1.with_columns(
            pl.col("hints").shuffle(seed)
        )

        # Seperate into psuedo groups for each condition
        x0 = shuffled_0.filter(
            pl.col("hints").str.contains("Standard Locking")
        )[col].to_numpy()
        y0 = shuffled_0.filter(
            pl.col("hints").str.contains("Lockahead")
        )[col].to_numpy()
        x1 = shuffled_1.filter(
            pl.col("hints").str.contains("Standard Locking")
        )[col].to_numpy()
        y1 = shuffled_1.filter(
            pl.col("hints").str.contains("Lockahead")
        )[col].to_numpy()

        # Compute effect of Lockahead under each condition
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


    # 4.1 Compute the Confidence interval for plotting
    centered_perm = perm_stats - np.mean(perm_stats) + observed
    ci_low, ci_high = np.percentile(centered_perm, [2.5, 97.5])


    return observed, (ci_low, ci_high), p_value


if __name__ == '__main__':
    dataframe = parse_data(sample_size=3500, verbose=False)
    results_frame = pl.DataFrame()

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
                observed, confs, p = permutation_did(control_no_cb, control_cb,
                                              raid_no_cb, raid_cb,
                                              n_perm=100)
                print(confs)
                result = pl.DataFrame([
                    pl.Series("observed", [observed], pl.Float64),
                    pl.Series("p", [p], pl.Float64),
                    pl.Series("ci_low", [confs[0]], pl.Float64),
                    pl.Series("ci_high", [confs[1]], pl.Float64),
                    pl.Series("striping", [striping], pl.String),
                    pl.Series("xios_nodes", [int(node)], pl.Int32),
                ])

                results_frame = results_frame.vstack(result)

    with pl.Config(tbl_rows=-1):
        print(results_frame)

    results_frame = results_frame.with_columns(
        ci_low_coord = pl.col("observed").sub(pl.col("ci_low")),
        ci_high_coord = pl.col("ci_high").sub(pl.col("observed"))
    )

    # Now do some plotting
    sns.pointplot(data=results_frame,
                x="xios_nodes",
                y="observed",
                hue="striping",
                markers='o',
                linestyles=''
                )

    # get yerrs
    yerr = np.vstack([
        pl.Series(results_frame.select(pl.col("ci_low_coord"))).to_list(),
        pl.Series(results_frame.select(pl.col("ci_high_coord"))).to_list()
    ])

    plt.errorbar(
        x=pl.Series(results_frame.select(pl.col("xios_nodes"))).to_list(),
        y=pl.Series(results_frame.select(pl.col("observed"))).to_list(),
        yerr=yerr,
        fmt="none",
        capsize=4,
        linewidth=1.5
    )


    plt.ylim(-1, 1)
    plt.ylabel("Observed Effect Size")
    plt.xlabel("XIOS Nodes")
    plt.title("Cliff's Delta Observed Effect Size")
    plt.show()