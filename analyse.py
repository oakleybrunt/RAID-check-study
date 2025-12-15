import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

plot = True
comp = False

nnodes = "two"
compnodes = "two"
striped = "striped/"

plots_of_interest = ['lfric_gl_model_levs_diags_1hr',
                     'lfric_gl_model_levs_diags_3hr',
                     'lfric_gl_std_levs_diags_1hr',
                     'lfric_gl_std_levs_diags_3hr',
                     'lfric_name',
                     'lfric_ppvar']

# Set global matplotlib styling
style = {'axes.facecolor': 'white',
         'axes.edgecolor': 'black',
         'axes.linewidth': 0.5,
         'xtick.bottom': True,
         'axes.spines.bottom': True,
         'axes.titlesize': 'xx-large',
         'axes.labelsize': 16,
         'ytick.left': True,
         'xtick.color': 'black',
         'ytick.color': 'black',
         'xtick.labelsize': 14,
         'ytick.labelsize': 14,
         'font.family': 'serif',
         'lines.linewidth': 1.6,
         'lines.linestyle': 'dashed',
         'legend.frameon': False,
         'legend.fontsize': 'large',
         'legend.title_fontsize': 'large',
         'figure.titlesize': 30,
         'figure.figsize': [20, 25]
        }
sns.set_theme(rc=style)

cb_data = pl.read_csv(f"data/{nnodes}_node_xios/{striped}coll_buff.csv", infer_schema_length=100000)
nh_data = pl.read_csv(f"data/{nnodes}_node_xios/{striped}no_hints.csv", infer_schema_length=100000)

# Convert date field to correct format and add identifier col
cb_data = cb_data.with_columns(
    pl.col("created").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"),
    pl.col("raw_write_rate_mibs").cast(pl.Float64, strict=False).alias("raw_write_rate_mibs"),
    Identifier = pl.lit(f"collective buffering hints ({nnodes} xios nodes)"),
    size_GiBs = pl.col("n_bytes").truediv(1024**3),
    index = pl.int_range(pl.len(), dtype=pl.UInt32)
    )
nh_data = nh_data.with_columns(
    pl.col("created").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"),
    pl.col("raw_write_rate_mibs").cast(pl.Float64, strict=False).alias("raw_write_rate_mibs"),
    Identifier = pl.lit(f"no hints ({nnodes} xios nodes)"),
    size_GiBs = pl.col("n_bytes").truediv(1024**3),
    index = pl.int_range(pl.len(), dtype=pl.UInt32)
    )

if comp:
    cb_data_unstriped = pl.read_csv(f"data/{compnodes}_node_xios/striped/coll_buff.csv", infer_schema_length=100000)
    nh_data_unstriped = pl.read_csv(f"data/{compnodes}_node_xios/striped/no_hints.csv", infer_schema_length=100000)
    cb_data_unstriped = cb_data_unstriped.with_columns(
        pl.col("created").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"),
        pl.col("raw_write_rate_mibs").cast(pl.Float64, strict=False).alias("raw_write_rate_mibs"),
        Identifier = pl.lit(f"collective buffering hints - STRIPED ({compnodes} xios nodes) "),
        size_GiBs = pl.col("n_bytes").truediv(1024**3),
        index = pl.int_range(pl.len(), dtype=pl.UInt32)
        )
    nh_data_unstriped = nh_data_unstriped.with_columns(
        pl.col("created").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"),
        pl.col("raw_write_rate_mibs").cast(pl.Float64, strict=False).alias("raw_write_rate_mibs"),
        Identifier = pl.lit(f"no hints - STRIPED ({compnodes} xios nodes)"),
        size_GiBs = pl.col("n_bytes").truediv(1024**3),
        index = pl.int_range(pl.len(), dtype=pl.UInt32)
        )

stats_nh_data = nh_data.select(
    pl.col("raw_write_rate_mibs"),
    pl.col("size_GiBs"),
    # pl.col("aggregators"),
    # pl.col("collective_writes")
)

stats_cb_data = cb_data.select(
    pl.col("raw_write_rate_mibs"),
    pl.col("size_GiBs"),
    # pl.col("aggregators"),
    # pl.col("collective_writes")
)

with pl.Config(tbl_cols=-1):
    print("Collective Buffering")
    print(stats_cb_data.describe())
    print("No Hints")
    print(stats_nh_data.describe())

# max_date_nh = nh_data.select(pl.col("created").max())['created'].to_list()[0]
# max_date_cb = cb_data.select(pl.col("created").max())['created'].to_list()[0]
# max_date_nh_us = nh_data_unstriped.select(pl.col("created").max())['created'].to_list()[0]
# max_date_cb_us = cb_data_unstriped.select(pl.col("created").max())['created'].to_list()[0]


max_date_nh = nh_data.select(pl.col("index").max())['index'].to_list()[0]
max_date_cb = cb_data.select(pl.col("index").max())['index'].to_list()[0]

if comp:
    max_date_nh_us = nh_data_unstriped.select(pl.col("index").max())['index'].to_list()[0]
    max_date_cb_us = cb_data_unstriped.select(pl.col("index").max())['index'].to_list()[0]

min_date = min(max_date_cb, max_date_nh)

if comp:
    min_date_us = min(max_date_nh_us, max_date_cb_us)
    min_date = min(min_date, min_date_us)

# with pl.Config(tbl_cols=-1):
#     print(nh_data)
#     print(cb_data)


# Combine the two datasets
plot_data = pl.concat([cb_data, nh_data], how="vertical_relaxed")

if comp:
    plot_data = pl.concat([plot_data, nh_data_unstriped], how="vertical_relaxed")
    plot_data = pl.concat([plot_data, cb_data_unstriped], how="vertical_relaxed")

plot_data = plot_data.filter(
    pl.col("index").le(min_date)
)

plot_data = plot_data.drop_nulls()


# Convert write rate to GiBs by dividing by 1024
plot_data = plot_data.with_columns(
    raw_write_rate_gibs = pl.col("raw_write_rate_mibs").truediv(1024)
)

# Get unique names
# names = plot_data.unique("filename")['filename'].to_list()


# with pl.Config(tbl_cols=-1):
#     print(plot_data)

RAID_END = datetime(year=2025, month=12, day=4, hour=15, minute=5)


if plot:
    for name in plots_of_interest:

        file_data = plot_data.filter(
            pl.col('filename').str.contains(name)
        )

        # Finally do some plotting
        # g = sns.jointplot(data=file_data,
        #             x='created',
        #             y='raw_write_rate_gibs',
        #             kind="scatter",
        #             hue="Identifier",
        #             height=15,
        #             )
        # g.ax_marg_x.remove()

        g = sns.jointplot(data=file_data,
                    x='index',
                    y='raw_write_rate_gibs',
                    kind="scatter",
                    hue="Identifier",
                    height=15,
                    )
        g.ax_marg_x.remove()

        plt.title(f"Comparative Write Rates for {name} with and without MPIIO Collective Buffering")
        plt.ylabel("Raw Write Rate (GiB/s)")
        # plt.ylabel("Collective Writes")
        # plt.xlabel("Model End Datetime")
        plt.xlabel("Run Number")

        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
        # plt.xticks(rotation=45)

        plt.ylim((-0.05, 10))
        plt.xlim((-50, min_date*1.1))
        g.figure.set_figwidth(22)
        g.figure.set_figheight(11)
        # plt.axvline(x=RAID_END, color="red", linewidth=2.5)
        # plt.show()
        # plt.savefig(f"plots/{striped}{nnodes}_nodes/{name}.png")
        plt.savefig(f"plots/{striped}{nnodes}_nodes/{name}.png")
        plt.close()
