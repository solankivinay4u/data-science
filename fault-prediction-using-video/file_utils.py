from operator import truediv
import shutil
from pathlib import Path
import os

def move_files(files_to_move, dest_path):
    for f in files_to_move:
        dest_file=os.path.join(dest_path,f.split('\\')[-1])
        if not os.path.isfile(dest_file):
        #if not Path(dest_path).joinpath(f.split()[-1]).exists():
            shutil.move(f,dest_path)
        else:
            #delete from src (duplicate file)
            os.remove(f)


def create_file(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as fp:
            pass

def is_dir_exist(dir_path):
    return Path(dir_path).is_dir()

def create_dir(dir_path):
    os.mkdir(dir_path)
    
def list_dirs(dir_path):
    return [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
def move_file(src_file, dest_path):
    shutil.move(src_file, dest_path)
    
def get_file_name_from_path(file_path):
    head, file_name= os.path.split(file_path)
    return file_name

def is_file_exist(file_path):
    return os.path.isfile(file_path)

def delete_file(file_path):
    if is_file_exist(file_path):
        os.remove(file_path)
        return True
    else:
        return False
    