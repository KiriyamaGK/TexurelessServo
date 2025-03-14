import h5py
from utils.paths import return_disc_route
from utils.file import ensure_dir
import os

if __name__ == '__main__':
    delete_demo_num=2

    base_dir = return_disc_route("One Touch")
    file_name="25.03.11"
    database_dir = os.path.join(base_dir, 'AlignAnything_real', file_name, 'hdf5')
    ensure_dir(database_dir)
    dataset_dir = os.path.join(database_dir, 'mimic.hdf5')
    new_f_out = h5py.File(dataset_dir, 'r+')

    if 'data' in new_f_out :
        existed_demo_num=len(new_f_out["data"])
        delete_demo_num = min(existed_demo_num,delete_demo_num)
        for uu in range(delete_demo_num):
            del new_f_out['data/demo_{}'.format(existed_demo_num - 1-uu)]