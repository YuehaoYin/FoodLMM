import os

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils.utils import is_main_process


class Config:
    def __init__(self, args):
        self.args = vars(args)
        self.running_path = os.getcwd()
        user_config = self._build_opt_list(args.options)
        config_file = os.path.join(self.running_path, 'train_config.yaml')
        if args.cfg_file is not None:
            config_file = args.cfg_file
        config = OmegaConf.load(config_file)
        self._merge_configs(config, **user_config)

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def _convert_to_dot_list(opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        has_equal = opts[0].find("=") != -1
        if has_equal:
            return opts
        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _merge_configs(self, config, **kwargs):
        self._recurse_config(config)
        for key in kwargs:
            self.args[key] = kwargs.get(key)

    def _recurse_config(self, config):
        for key, value in config.items():
            if isinstance(value, DictConfig):
                self._recurse_config(value)
            else:
                self.args[key] = value

    def pretty_print_system(self):
        if is_main_process():
            print(f"\n======  Configuration  ======")
            for key in self.args:
                print('{} = {}'.format(key, self.args[key]))
            print(f"=============================")
