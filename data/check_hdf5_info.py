import h5py
from utils.paths import PROJECT_ROOT_DIR

if __name__ == '__main__':
    hdf_date = "25.10.01"
    hdf_fn = "mimic.hdf5"
    hdf_pth = f"/media/kiriyamagk/One Touch/AlignAnything/{hdf_date}/hdf5/{hdf_fn}"

    f = h5py.File(hdf_pth, "r")
    to_check = f["data"].attrs["env_args"]
    print(to_check)
