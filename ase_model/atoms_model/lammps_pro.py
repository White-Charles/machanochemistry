# packages

import glob
import os
import numpy as np
import pandas as pd

from ase import Atoms
from os.path import join
from ase.io import read
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from sympy import N
from atoms_model import sort_z
from pathlib import Path

def extract_energy(filepath):
    """从 LAMMPS 日志文件中提取能量值"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"警告: 文件未找到 {filepath}")
        return None

    for i, line in enumerate(lines):
        if "next-to-last" in line:
            # 获取匹配行后的三行
            next_lines = lines[i+1:i+4]
            if len(next_lines) < 3:
                continue
            # 取这三行中的第一行
            target_line = next_lines[0].split()
            if len(target_line) >= 3:
                try:
                    return float(target_line[2])
                except ValueError:
                    continue
    print(f"警告: 无法在 {filepath} 中找到能量值")
    return None

def get_md_neb_result(result_path, neb_path='results'):
    """处理指定路径下的NEB计算结果"""
    # 获取并排序所有日志文件
    pattern = os.path.join(result_path, "log.lammps.*")
    files = sorted(glob.glob(pattern),
                  key=lambda x: int(x.split('.')[-1]))

    if not files:
        print(f"错误: 在 {result_path} 中未找到 log.lammps.* 文件")
        return

    # 提取初始构象能量
    E0 = extract_energy(files[0])
    if E0 is None:
        print(f"错误: 无法从 {files[0]} 提取初始能量")
        return

    num_files = len(files)

    # 提取路径
    neb_models = get_dump_files(join(result_path,neb_path))
    neb_dist = get_dist_list(neb_models)
    # 打印表头
    print("reaction_coordinate de")
    print(f"0 0")  # 初始点
    neb_x = [.0]
    neb_e = [.0]
    # 处理中间构象
    for i in range(1, num_files):
        E = extract_energy(files[i])
        if E is None:
            continue

        # 计算能量差和反应坐标
        de = E - E0
        rc = neb_dist[i] / neb_dist[-1]  # 反应坐标

        # 格式化输出
        print(f"{rc:.3f} {de:.3f}")
        neb_x.append(rc)
        neb_e.append(de)
    return(np.array(neb_x), np.array(neb_e))

def get_location(atoms_model, order):
    # 获取指定位置的原子坐标
    order_x = order[0]
    order_y = order[1]
    order_z = order[-1]
    # 先定位其层数
    pos = atoms_model.get_positions()
    num = atoms_model.get_global_number_of_atoms()
    sortz = sort_z(pos[:,-1])
    order_z = list(range(int(max(sortz)+1)))[order_z]
    z_num = np.arange(num)[sortz==order_z]
    pos_z = pos[z_num,:]
    sortx = sort_z(pos_z[:,0])
    order_x = list(range(int(max(sortx)+1)))[order_x]
    x_num = z_num[sortx == order_x]
    pos_x = np.array(pos[x_num])
    sorty = sort_z(pos_x[:,1])
    order_y = list(range(int(max(sortx)+1)))[order_y]
    y_num = x_num[sorty == order_y]
    pos_select = pos[y_num][0]
    return y_num, pos_select


def read_lammps_result(path, dump_fielname='result.dat', style='charge'):
    """读取 LAMMPS 数据文件"""
    filename = join(path,dump_fielname)
    result = read_lammps_data(filename, style=style)
    return result


def read_lammps_onedata(path,  dump_fielname='result.dat', format='lammps-data', style='charge'):
    """读取 LAMMPS 数据文件"""
    if format == 'lammps-data':
        result = read_lammps_result(path, dump_fielname=dump_fielname, style=style)
    elif format == 'lammps-dump-text':
        result = read(join(path, dump_fielname))

    # result.get
    symbols = set(result.get_chemical_symbols())
    if len(symbols) == 1:
        symbol_map={'H':'Pt'}
    elif len(symbols) == 2:
        symbol_map={'H': 'O', 'He':'Pt'}
    elif len(symbols) == 3:
        symbol_map={'He': 'O', 'Li':'Pt'}
    else:
        raise ValueError("symbols length > 3")
    result = remap_atoms(result, symbol_map=symbol_map)

    return result


def extract_last_toteng(path, filename='log.lammps'):
    """
    从 LAMMPS 输出文件中提取最后一步的 TotEng 值

    参数:
        path (str): LAMMPS 输出文件路径
        filename (str): LAMMPS 输出文件名，默认为 'log.lammps'
    返回:
        float: 最后一步的总能量值 (TotEng)
    """
    # 读取整个文件
    filename = join(path, filename)
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 寻找包含表头的行
    header_line = None
    for i, line in enumerate(lines):
        if "TotEng" in line:
            header_line = i
            break

    if header_line is None:
        raise ValueError("Can't find the 'TotEng' table header in the file")

    # 提取表头并清理
    headers = lines[header_line].split()
    # 收集数据行
    data_lines = []
    for line in lines[header_line+1:]:
        # 跳过空行和注释行
        if not line.strip() or line.startswith("#") or line.startswith("Loop"):
            break

        # 检查是否为有效数据行
        if len(line.split()) == len(headers):
            data_lines.append(line)

    # 创建 DataFrame
    if not data_lines:
        raise ValueError("在表头后未找到有效数据行")

    # 创建 DataFrame
    df = pd.DataFrame([line.split() for line in data_lines], columns=headers)

    # 将数值列转换为浮点数
    for col in df.columns[1:]:  # 跳过 Step 列
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # print(df)
    # 提取最后一行
    last_row = df.iloc[-1]

    # 返回 TotEng 值
    return last_row["TotEng"]


def remap_atoms(atoms, symbol_map):
    """
    根据映射关系重设原子符号

    参数:
        atoms (ase.Atoms): 原始原子结构
        symbol_map (dict): 符号映射字典 {原始符号: 目标符号}

    返回:
        ase.Atoms: 映射后的新原子结构
    """
    # 获取原始符号列表
    original_symbols = atoms.get_chemical_symbols()

    # 创建映射后的符号列表
    remapped_symbols = []

    for symbol in original_symbols:
        # 应用映射关系
        if symbol in symbol_map:
            remapped_symbols.append(symbol_map[symbol])
        else:
            # 对于未指定的符号，保留原样
            remapped_symbols.append(symbol)
    # 创建新的Atoms对象
    new_atoms = atoms.copy()
    new_atoms.set_chemical_symbols(remapped_symbols)

    return new_atoms

def get_md_e(result_path, energy_file="log.lammps"):
    """处理指定路径下的lammps计算结果能量"""
    # 提取初始构象能量
    E0 = extract_energy(join(result_path,energy_file))
    if E0 is None:
        print(f"错误: 无法从 {result_path} 提取初始能量")

    return(E0)

def get_dump_files(directory, suffix='.text'):
    """
    查找指定目录及其子目录中所有后缀为 .text 的文件

    参数:
        directory (str): 要扫描的根目录路径

    返回:
        list: 包含所有 .text 文件完整路径的列表
    """
    # 确保目录存在
    if not os.path.exists(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")


    # 使用 pathlib 进行高效遍历
    text_files = []
    dir_path = Path(directory)

    # 递归遍历所有文件
    for file_path in dir_path.rglob('*'):
        # 检查是否为文件且后缀为 .text (不区分大小写)
        if file_path.is_file() and file_path.suffix.lower() == suffix:
            text_files.append(str(file_path.resolve()))

    models= []
    for file in text_files:
        print(file)
        # print(file)
        result = read(file, format='lammps-dump-text')
        symbols = set(result.get_chemical_symbols())
        if len(symbols) == 1:
            symbol_map={'H':'Pt'}
        elif len(symbols) == 2:
            symbol_map={'H': 'O', 'He':'Pt'}
        elif len(symbols) == 3:
            symbol_map={'He': 'O', 'Li':'Pt'}
        else:
            raise ValueError("symbols length > 3")
        result = remap_atoms(result, symbol_map=symbol_map)
        models.append(result)
    return models

from ase.geometry import conditional_find_mic

def get_pointwise_distances(p1, p2, cell=None, pbc=None):
    """Return displacement vectors and distances between corresponding points in p1 and p2

    Parameters:
    p1, p2: Arrays of points with shape (N, 3)
    cell: Unit cell vectors for periodic boundary conditions (3x3 matrix)
    pbc: Periodic boundary conditions flags (length 3 boolean array)

    Returns:
    D: Displacement vectors (p2 - p1) with shape (N, 3)
    D_len: Distances with shape (N,)
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)

    # 确保输入点集大小一致
    if len(p1) != len(p2):
        raise ValueError("p1 and p2 must contain the same number of points")

    # 直接计算对应点间的位移向量
    D = p2 - p1

    # 应用周期性边界条件
    (D, ), (D_len, ) = conditional_find_mic([D], cell=cell, pbc=pbc)

    return D, D_len

def get_displacement(atoms1, atoms2):
    # 计算两个构型的距离
    cell = atoms1.cell
    pbc = atoms1.pbc
    _, D_len = get_pointwise_distances(atoms1.positions, atoms2.positions, cell=cell, pbc=pbc)
    # print(D_len)
    # diagonal_vectors = D_len.diagonal()
    # print(diagonal_vectors)
    magnitudes = np.linalg.norm(D_len, axis=0)

    return(magnitudes)

def get_dist_list(neb_list):
    # 计算一个neb
    dist_list = []
    for i in range(len(neb_list)):
        j = max(i-1, 0)
        d = get_displacement(neb_list[j], neb_list[i])
        dist_list.append(d)

    dist_list = [sum(dist_list[:i+1]) for i in range(len(dist_list))]
    return(np.array(dist_list))


def write_lammps_onedata(path, model, style='charge'): # 定制LAMMPS数据输出函数
    """将模型写入 LAMMPS 数据文件"""
    filename = join(path, 'lammps.data')
    write_lammps_data(filename, model, atom_style=style)


def write_refer_atoms(models, outpath='./ads.data', ads_num=None, slab_symbels=['Pt']):
    if ads_num is None:
        ads_num = [i for i, atom in enumerate(models) if atom.symbol not in slab_symbels]
    else:
        ads_num = [range(len(models))[i] for i in ads_num]
    ads_model = models[ads_num]
    ads_pos = ads_model.get_positions()
    ads_num = list(np.array(ads_num).reshape(-1,1)+1)
    data = np.hstack((ads_num, ads_pos))
    # 写入txt文件
    with open(outpath, 'w') as f:
        # 第一行写入原子数量
        f.write(f"{len(ads_num)}\n")

        # 写入每个原子的索引和坐标
        for row in data:
            # 格式化输出：索引占7位，每个坐标占24位（保留15位小数）
            f.write(f"{int(row[0]):7d}{row[1]:24.15f}{row[2]:24.15f}{row[3]:24.15f}\n")



from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NEB_path_e:
    """存储单个路径的能量剖面数据"""
    name: str
    path_points: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_values: np.ndarray = field(default_factory=lambda: np.array([]))
    models: Optional[List[Atoms]] = field(default_factory=list)  # 可选的原子结构列表
    strain: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
    strain_value: Optional[float] = None
    strain_symbel: Optional[str] = None

    @property
    def relative_energy(self) -> np.ndarray:
        """计算相对于第一个点的相对能量"""
        if len(self.energy_values) == 0:
            return np.array([])
        return self.energy_values - self.energy_values[0]

    def add_point(self, path: float, energy: float, model: Atoms = None):
        """添加一个路径点及其对应的能量值"""
        self.path_points = np.append(self.path_points, path)
        self.energy_values = np.append(self.energy_values, energy)
        if model is not None:
            self.models.append(model)

    def get_max_energy(self) -> tuple[float, float]:
        """返回最大能量值及其对应的路径点"""
        if len(self.energy_values) == 0:
            return (0.0, 0.0)
        max_index = np.argmax(self.energy_values)
        return (self.path_points[max_index], self.energy_values[max_index])

    def get_is_barrier(self) -> tuple[float, float]:
        """返回正向的势垒"""
        if len(self.energy_values) == 0:
            return (0.0, 0.0)
        max_index = np.argmax(self.energy_values)
        return (self.energy_values[max_index]-self.energy_values[0])

    def get_fs_barrier(self) -> tuple[float, float]:
        """返回逆向的势垒"""
        if len(self.energy_values) == 0:
            return (0.0, 0.0)
        max_index = np.argmax(self.energy_values)
        return (self.energy_values[max_index]-self.energy_values[-1])

    def __repr__(self) -> str:
        is_barrier = self.get_is_barrier() if len(self.energy_values) > 0 else 0.0
        fs_barrier = self.get_fs_barrier() if len(self.energy_values) > 0 else 0.0

        models_count = len(self.models) if self.models is not None else 0
        return (f"<EnergyProfile: {self.name} with {len(self.path_points)} points,{models_count} models,  "
                f"is barrier: {is_barrier:.3f},fs barrier: {fs_barrier:.3f}>"
                )

    def __post_init__(self):
        """确保数据是 NumPy 数组"""
        if not isinstance(self.path_points, np.ndarray):
            self.path_points = np.array(self.path_points)
        if not isinstance(self.energy_values, np.ndarray):
            self.energy_values = np.array(self.energy_values)
        # 确保models总是列表（即使传入None）
        if self.models is None:
            self.models = []