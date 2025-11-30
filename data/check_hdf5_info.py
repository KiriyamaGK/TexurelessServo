import h5py
from utils.paths import PROJECT_ROOT_DIR
import json

if __name__ == '__main__':
    hdf_date = "25.11.20"
    hdf_fn = "mimic.hdf5"
    hdf_pth = f"/media/kiriyamagk/One Touch/AlignAnything_real/{hdf_date}/hdf5/{hdf_fn}"
    _do_save = True

    f = h5py.File(hdf_pth, "r")
    to_check = f["data"].attrs["env_args"]
    print(to_check)
    if _do_save:
        parsed_data = json.loads(to_check)  # 字符串转对象
        with open(hdf_date + ".json", "w") as jf:
            json.dump(parsed_data, jf, indent=4)
        print("Config saved successfully.")
    f.close()