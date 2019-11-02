import yaml
import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(config_file_path):
    with open(config_file_path) as f:
        params = yaml.load(f.read())
        # We expand ~ in those yaml entries with `path` on their keys for making
        # config files more platform-independent
        params = {key: (os.path.expanduser(value) if 'path' in key and value is not None
                        else value) for key, value in params.items()}
        params = AttrDict(params)
    return params