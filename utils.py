
import os
from urllib.request import urlopen
import tarfile
import pickle

PICKLE_PATH = 'data/'
RAW_DATA_PICKLE = 'raw_data.pickle'
NORM_DATA_PICKLE = 'normalized_data.pickle'


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


def save_object(object, object_name, data_path=PICKLE_PATH):
  with open(os.path.join(data_path, object_name), 'wb') as f:
    pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
  return object

def load_object(object_name, data_path=PICKLE_PATH):
  with open(os.path.join(data_path, object_name), 'rb') as f:
    object = pickle.load(f)
  return object



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

def load_data(data_path):
  if os.path.isfile(os.path.join(PICKLE_PATH, RAW_DATA_PICKLE)):
    return load_object(RAW_DATA_PICKLE)
  if not os.path.isdir(data_path):
    donwload_data()
  return save_object(get_data(data_path), RAW_DATA_PICKLE)

X_train, y_train, X_test, y_test = get_data('data/')