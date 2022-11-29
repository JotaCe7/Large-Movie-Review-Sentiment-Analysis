
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
  """
  Save an object using pickle library
  Paramenters
  -----------
  object : object
      Object to be saved
  object_name : str
      Name used to save object
  data_pathh : str
      relative/absolute path where object will be saved
  
  Returns
  -------
      Returns (byspass) the object to be saved
  """
  with open(os.path.join(data_path, object_name), 'wb') as f:
    pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
  return object

def load_object(object_name, data_path=PICKLE_PATH):
  """
  Load an object which was previously saved using pickle library
  Paramenters
  -----------
  object_name : str
      Name of the saved obbject
  data_pathh : str
      relative/absolute path where object will is saved
  
  Returns
  -------
      Loaded object
  """
  with open(os.path.join(data_path, object_name), 'rb') as f:
    object = pickle.load(f)
  return object



def donwload_data(url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
  """
  Donwload uncompressed needed files from url dataset
  
  Parameters
  ----------
  url : str
      url where to download the dataset from
  """
  with urlopen(url) as tgz:
      with tarfile.open(fileobj=tgz, mode="r|gz") as tgz:
          for file in tgz:
            print(file.name)
            if file.isfile() & ((('/neg' in file.name) or ('/pos' in file.name)) or ('/unsup' in file.name)):
                tgz.extract(file)

def get_data(data_path):
  """
  Get splitted dataset features and labels

  Parameters
  ----------
  datapath: str
      Path where dataset is stored.
  
  Returns
  -------
      [X_train, y_train, X_test, y_test] A list containing training set
      and test set featurtes  and labels
  """
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
  """
  Load dataset in case it was previously downloaded. If not donwloads it
  and saves it
  
  Parameters
  ----------
  datapath : str
      Path where dataset is stored or will be donwloaded.

  Returns
  -------
      D
  """
  if os.path.isfile(os.path.join(PICKLE_PATH, RAW_DATA_PICKLE)):
    return load_object(RAW_DATA_PICKLE)
  if not os.path.isdir(data_path):
    donwload_data()
  return save_object(get_data(data_path), RAW_DATA_PICKLE)

X_train, y_train, X_test, y_test = get_data('data/')