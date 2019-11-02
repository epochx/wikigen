
import os
from collections import namedtuple
import warnings
import json

from .diff import PatchSet


def read_diff_file(diff_file_path):
    with open(diff_file_path) as diff_file:
        diff = diff_file.read()#.decode('utf8', errors='ignore')
        diff_data = PatchSet(diff.splitlines())
    return diff_data


def read_message_file(message_file_path):
    with open(message_file_path, 'r') as f:
        content = f.read()#.decode('utf-8', errors='ignore')
        content = content.strip()

    output = {'commit': {'message': content}}

    return output

Commit = namedtuple('Commit', ['sha', 'metadata', 'diff'], verbose=False)


class WikigenCorpus(object):
    """
    Lazy loader
    """

    def __init__(self, data_path):

        self._json_path = os.path.join(data_path, "txt")
        self._diffs_path = os.path.join(data_path, "diff")

        diff_files = os.listdir(self._diffs_path)
        json_files = os.listdir(self._json_path)

        shas_diff = [f.replace('.diff', '') for f in diff_files]
        shas_json = [f.replace('.txt', '') for f in json_files]

        if not set(shas_diff) == set(shas_json):
            warnings.warn("There were missing files")
            self.shas = list(set(shas_diff) & set(shas_json))
        else:
            self.shas = shas_json

    def __len__(self):
        return len(self.shas)

    def __getitem__(self, index):
        sha = self.shas[index]

        diff_filepath = os.path.join(self._diffs_path, sha + '.diff')
        try:
            with open(diff_filepath, 'r') as diff_file:
                diff = diff_file.read()
                diff_data = PatchSet(diff.splitlines())

            json_filepath = os.path.join(self._json_path, sha + '.txt')
            with open(json_filepath, 'r') as json_file:
                json_data = json.load(json_file)

            commit = Commit(sha, json_data, diff_data)

            return commit

        except Exception as e:
            warnings.warn(str(e) + " in sha " + sha)
            return None
