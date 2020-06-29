import os
import tarfile

import config
#from .config import DATA_DIR, aishell_folder

def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall(config.DATA_DIR)
    tar.close()

if __name__ == "__main__":
    if not os.path.isdir(config.aishell_folder):
        extract(config.DATA_DIR+'/data_aishell.tgz')
