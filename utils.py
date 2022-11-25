
import os
from urllib.request import urlopen
import tarfile

def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def donwload_data(url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
  with urlopen(url) as tgz:
      with tarfile.open(fileobj=tgz, mode="r|gz") as tgz:
          for file in tgz:
            print(file.name)
            if file.isfile() & (('/neg' in file.name) or ('/pos' in file.name)):
                tgz.extract(file)


def get_data(data_path):
  result = []
  target_dict = {'pos':1, 'neg':0}
  for split in ['train', 'test']:
    X , y = [], []
    for target in target_dict:
      for path in walkdir(os.path.join(data_path,split,target)):
        with open(os.path.join(*path)) as f:
          X.append(f.read())
        y.append(target_dict[target])
    result.extend([X,y])
  return result
X_train, y_train, X_test, y_test = get_data('data/')