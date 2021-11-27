import os
import json
import shutil
from pathlib import Path

def to_path(path):
    p = path if isinstance(path, Path) else Path(path)
    return p

def dirs(path, glob='*'):
    p = to_path(path)
    return [str(x) for x in p.glob(glob) if x.is_dir()]

def files(path, glob='*'):
    p = to_path(path)
    return [str(x) for x in p.glob(glob) if x.is_file()]

def join(*argv):
    return str(Path(*argv))

def join_and_create_dir(*argv):
    p = join(*argv)
    create_dir(p)
    return str(p)

def exists(path):
    return os.path.exists(path)

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            return path
    except Exception as e:
        print(e)

def stem(path):
    p = to_path(path)
    return p.stem

def basename(path):
    return os.path.basename(path)

def json_dump(path: str, data, indent=4):
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def json_load(path: str):
    data = {}

    with open(path) as f:
        data = json.load(f)
    return data

def read_file(path: str):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    return lines

def write_file(path: str, lines):
    file = open(path, 'w')
    file.writelines(lines)
    file.close()

def cp(src, dst):
    shutil.copy(src, dst)


def group_buckets(items, key, value=None):
    buckets = {}
    for x in items:
        k = key(x)
        if k not in buckets:
            buckets[k] = []
        if not value:
            buckets[k].append(x)
        else:
            v = value(x)
            buckets[k].append(v)
    return buckets