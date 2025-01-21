from utils.hdf5 import split_train_val_from_hdf5,copy_attributes
import h5py
import os
import numpy as np



if __name__ == "__main__":
    folder_path_1 = '/media/kiriyamagk/One Touch/AlignAnything/25.01.18/hdf5/mimic.hdf5'
    folder_path_2 = '/media/kiriyamagk/One Touch/AlignAnything/25.01.19/hdf5/mimic.hdf5'
    merged_path='/media/kiriyamagk/One Touch/AlignAnything/25.01.19/hdf5/merged.hdf5'
    val_ratio = 0.1

    # List all hdf5 files in the directory
    hdf5_files = [folder_path_1, folder_path_2]

    counter = 0

    # Create or open the merged.hdf5 file
    with h5py.File(merged_path, "w") as merged_file:
        # Create a group named data if it doesn't exist yet
        data_group = merged_file.require_group("data")

        # Iterate over all the hdf5 files and merge demos
        for hdf5_file in hdf5_files:
            print("merging {}".format(hdf5_file))
            with h5py.File(hdf5_file, 'r') as source_file:
                source_data_group = source_file['data']
                if counter == 0:
                    copy_attributes(source_data_group, data_group)

                # Iterate through all demos in the 'data' group of the source file
                for demo_name in source_data_group:
                    print('merging {}'.format(demo_name))
                    new_demo_name = f"demo_{counter}"

                    # Copy demo to merged_file
                    source_file.copy(f"data/{demo_name}", data_group, new_demo_name)

                    # Copy attributes of the demo dataset
                    copy_attributes(source_data_group[demo_name], data_group[new_demo_name])

                    counter += 1
    print("Merging completed!")

    # store metadata about number of samples
    total_samples = 0
    f = h5py.File(merged_path, "a") # edit mode
    for ep in f["data"]:

        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0]-1
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

        #print("num_samples:",n_sample)
    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples
    #print("totol_samples:", total_samples)

    #spilt
    split_train_val_from_hdf5(merged_path, val_ratio, filter_key=None)