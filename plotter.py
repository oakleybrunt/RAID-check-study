import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

from config import Config
from csv_parser import parse_data


def set_style():
    # Set global matplotlib styling
    style = {'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.5,
            'xtick.bottom': True,
            'axes.spines.bottom': True,
            'axes.titlesize': 'large',
            'axes.labelsize': 16,
            'ytick.left': True,
            'xtick.color': 'black',
            'ytick.color': 'black',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'font.family': 'serif',
            'lines.linewidth': 1.6,
            'lines.linestyle': 'solid',
            'legend.frameon': False,
            'legend.fontsize': 'large',
            'legend.title_fontsize': 'large',
            'figure.titlesize': 30,
            # 'figure.figsize': [20, 25]
            }
    sns.set_theme(rc=style)


def plot_distributions(data=[], x=None, col=None, row=None, hue=None):
    if isinstance(data, pl.DataFrame):
        plot_data = [data]
    elif isinstance(data, list) and all(isinstance(entry, pl.DataFrame) for entry in data):
        plot_data = data
    else:
        raise Exception("data must be a pl.DataFrame or a list of pl.DataFrame objects")

    for curr_data in plot_data:
        g = sns.FacetGrid(curr_data, col=col,  row=row, hue=hue)
        g.map_dataframe(sns.histplot, x=x)
        plt.show()


if __name__ == "__main__":
    dataframe = parse_data(sample_size=3500, verbose=False)

    grouped_frame = dataframe.with_columns(
        group = pl.concat_str([pl.col("hints"),
                            pl.col('raid_level')],
                            separator=', ')
                            )

    striped_dataframe = grouped_frame.filter(
        pl.col("striping") == 'striped'
    )

    unstriped_dataframe = grouped_frame.filter(
        pl.col("striping") == 'unstriped'
    )

    set_style()

    # for node in [1, 2, 4]:
    #     plot_data = unstriped_dataframe.filter(
    #         pl.col("xios_nodes") == node
    #     )

    plot_data = unstriped_dataframe
    fig, ax = plt.subplots(nrows=1, ncols=2)

    vi = sns.violinplot(data=plot_data,
            x='xios_nodes',
            y='raw_write_rate_gibs',
            hue='group',
            hue_order=['Standard Locking, control',
                    'Standard Locking, RAID',
                    'Lockahead, control',
                    'Lockahead, RAID',
                    ],
            palette=Config.get().colours,
            alpha=0.7,
            ax=ax[1],
            )

    ec = sns.ecdfplot(data=plot_data,
                x='raw_write_rate_gibs',
                hue='group',
                hue_order=[
                    'Standard Locking, control',
                    'Standard Locking, RAID',
                    'Lockahead, control',
                    'Lockahead, RAID',
                    ],
                palette=Config.get().colours,
                ax=ax[0]
                )

    fig.set_figwidth(15.5)
    fig.set_figheight(6)
    ax[1].set_ylim(-1, 20)
    ax[1].set_ylabel("Raw Write Rate (GiB/s)")
    ax[1].set_xlabel("XIOS Nodes")
    ax[0].set_xlabel("Raw Write Rate (GiB/s)")
    ax[0].set_ylabel("Cumulative Proportion of Runs")
    # fig.suptitle(f"{node} XIOS nodes with unstriped output")
    fig.suptitle("Unstriped output")
    plt.savefig(f'paper_plots/UNSTRIPED_plot.png', dpi=500)
    plt.close()

    # for node in [2, 4]:
    #     plot_data = striped_dataframe.filter(
    #         pl.col("xios_nodes") == node
    #     )

    plot_data = striped_dataframe
    fig, ax = plt.subplots(nrows=1, ncols=2)

    sns.violinplot(data=plot_data,
            x='xios_nodes',
            y='raw_write_rate_gibs',
            hue='group',
            hue_order=['Standard Locking, control',
                    'Standard Locking, RAID',
                    'Lockahead, control',
                    'Lockahead, RAID',
                    ],
            palette=Config.get().colours,
            alpha=0.7,
            ax=ax[1],
            )

    sns.ecdfplot(data=plot_data,
                x='raw_write_rate_gibs',
                hue='group',
                hue_order=[
                    'Standard Locking, control',
                    'Standard Locking, RAID',
                    'Lockahead, control',
                    'Lockahead, RAID',
                    ],
                palette=Config.get().colours,
                ax=ax[0]
                )

    fig.set_figwidth(15.5)
    fig.set_figheight(6)
    ax[1].set_ylim(-1, 20)
    ax[1].set_ylabel("Raw Write Rate (GiB/s)")
    ax[1].set_xlabel("XIOS Nodes")
    ax[0].set_xlabel("Raw Write Rate (GiB/s)")
    ax[0].set_ylabel("Cumulative Proportion of Run")
    # fig.suptitle(f"{node} XIOS nodes with striped output")
    fig.suptitle("Striped output")
    plt.savefig(f'paper_plots/STRIPED_plot.png', dpi=500)
    plt.close()


__all__ = [
    'plot_distribution',
    'set_Style'
]