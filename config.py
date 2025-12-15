from datetime import datetime
import seaborn as sns

class Config():
    _instance = None

    @staticmethod
    def get():
        '''Static function that if necessary creates and returns the singleton
        config instance.

        '''
        if not Config._instance:
            Config._instance = Config()
        return Config._instance

    def __init__(self):
        Config.striping = ['striped', 'unstriped']
        Config.nodes = ['1', '2', '4']
        Config.raid_check = ['RAID', 'control']
        Config.filenames = ['coll_buff', 'no_hints']

        # RAID windows
        Config.RAID_START = datetime(year=2025, month=12, day=1, hour=1, minute=0)
        Config.RAID_END = datetime(year=2025, month=12, day=4, hour=15, minute=5)
        Config.RAID_START_2 = datetime(year=2025, month=12, day=11, hour=17, minute=0)


        coll_buff_raid = "#e61d1d"
        coll_buff_control = "#d86100"
        hints_raid = "#3b4ce2"
        no_hints_control = "#5d20ac"

        # Make palettes
        Config.colours = sns.color_palette([coll_buff_raid,
                                            coll_buff_control,
                                            hints_raid,
                                            no_hints_control,
                                        ])