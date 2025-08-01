import os
import shutil
import numpy as np
from send2trash import send2trash

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # 如果文件夹存在，则递归删除
    os.makedirs(folder_path)  # 创建空文件夹

def create_folder(folder_path, is_delete=False):
    if os.path.exists(folder_path):
        # 如果文件夹存在，则删除文件夹及其内部文件
        if is_delete:
            print(is_delete)
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                except:
                    send2trash(folder_path)
                print(f"Folder {folder_path} already exists.")
            print(f"delete {folder_path}")
            exist_folder(folder_path, is_create=True)
        else:
            print(f"folder {folder_path} exists")
    else:
        exist_folder(folder_path, is_create=True)

def del_file(file_path):
    """_summary_
        Delete the file if it exists, otherwise print a message.
    """
    # 指定要删除的文件路径
    # 尝试删除文件
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"The file {file_path} was successfully deleted ")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
    else:
        print(f" {file_path} does not exist")


def exist_folder(judge_folder_path, is_create=False):
    """_summary_
        Check if the folder exists. If the folder is wanted to be created,
        'is_create' should be seted to True.
    Args:
        judge_folder_path (_type_): _description_
        is_create (bool, optional): _description_. Defaults to False.

    Raises:
        FileNotFoundError: _description_
    """
    if os.path.exists(judge_folder_path):
        print(f"folder '{judge_folder_path}' exists")
    elif is_create:
        os.makedirs(judge_folder_path)
        print(f"folder '{judge_folder_path}' is created")
    else:
        raise FileNotFoundError(f"folder '{judge_folder_path}' does not exist, make sure the calculation has ended")


def cal_d(energies, densities, dband=False, darea=False):
    # 如果dband那么只输出 center

    if densities.ndim > 1:
        densities_arr = np.sum(densities, axis=np.argmin(densities.shape))
    else:
        densities_arr = np.array(densities)

    energies_arr = np.array(energies)
    # densities_arr = np.array(densities)
    area = np.trapz(densities_arr, energies_arr)
    mean_e = np.trapz(energies_arr * densities_arr, energies_arr/ area )
    if dband and not darea:
        return  mean_e
    elif darea:
        return area, mean_e
    else:
        var_e = np.trapz((energies_arr - mean_e)**2 * densities_arr/area, energies_arr)
        s = np.trapz((energies_arr - mean_e)**3 * densities_arr/area, energies_arr) / (var_e**(3/2))
        k = np.trapz((energies_arr - mean_e)**4 * densities_arr/area, energies_arr) / (var_e**2)
        return area, mean_e, var_e, s, k-3