import yaml
from yaml.loader import SafeLoader


def get_config(fpath="config.yml"):
    with open(fpath) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data
