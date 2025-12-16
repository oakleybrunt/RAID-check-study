import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config

from plotter import set_style


did_results_frame = pl.read_csv("did_results_frame.csv")
single_perm_results = pl.read_csv("single_perm_results.csv")
lock_mode_effect = pl.read_csv("locking_effect_sizes.csv")



# Now do some plotting
set_style()

plt.figure(figsize=(12,6))
sns.barplot(data=did_results_frame,
            x="xios_nodes",
            y="observed",
            hue="striping",
            palette=Config.get().colours,
            alpha=0.7
            )

plt.axhline(0, color='black', linewidth=0.4)
plt.ylim(-1, 1)
plt.ylabel("Observed Effect Size")
plt.xlabel("XIOS Nodes")
plt.title("Difference-in-differences Effect of Lock mode on Raw Write Rate")
plt.savefig("paper_plots/DiD_effect_plot.png", dpi=500)
plt.close()


sing_df = single_perm_results.with_columns(
    group = pl.concat_str([pl.col("striping"),
                        pl.col('lock_mode')],
                        separator=', ')
                        )

plt.figure(figsize=(12,6))
sns.barplot(data=sing_df,
            x="xios_nodes",
            y="observed",
            hue="group",
            palette=Config.get().colours,
            alpha=0.7,
            )

plt.axhline(0, color='black', linewidth=0.4)
plt.ylim(-1, 1)
plt.ylabel("Observed Effect Size")
plt.xlabel("XIOS Nodes")
plt.title("Effect Size of RAID on Raw Write Rate")
plt.savefig("paper_plots/RAID_effect_plot.png", dpi=500)
plt.close()


lock_df = lock_mode_effect.with_columns(
    group = pl.concat_str([pl.col("striping"),
                        pl.col('raid_level')],
                        separator=', ')
                        )

plt.figure(figsize=(12,6))
sns.barplot(data=lock_df,
            x="xios_nodes",
            y="observed",
            hue="group",
            palette=Config.get().colours,
            alpha=0.7
            )

plt.axhline(0, color='black', linewidth=0.4)
plt.ylim(-1, 1)
plt.ylabel("Observed Effect Size")
plt.xlabel("XIOS Nodes")
plt.title("Effect Size of Lockahead on Raw Write Rate")
plt.savefig("paper_plots/lock_effect_plot.png", dpi=500)
plt.close()