import os
import shutil

if __name__ == '__main__':
    data_dir = "D:\\Data\\Brain"
    dest_dir = "D:\\Data\\SSL4N\\T2"

    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if name.endswith(("_t2.nii.gz", "_T2w.nii.gz", "-T2.nii.gz")):
                file = root+"\\"+name
                #print(file)
                shutil.copy(file, dest_dir)

    print("Done!")