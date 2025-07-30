import os
import math
import shutil
import numpy as np

from lammps import lammps
from ase.atoms import Atoms
from ase.build import add_vacuum
from ase.calculators.emt import EMT
from ase.optimize import LBFGS, BFGS
from ase.io import write, read, lammpsdata
from ase.constraints import FixAtoms, FixCartesian
from operator import itemgetter


import matplotlib.pyplot as plt

from ase.neighborlist import natural_cutoffs, NeighborList
from ase.visualize.plot import Matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from math import sin, cos

def move_to_mid(slab):
    if isinstance(slab, Atoms):
        atom = slab.copy()
        pos = atom.get_scaled_positions()
        dis = 0.5 - np.average(pos[:,2])
        pos[:,2] += dis
        atom.set_scaled_positions(pos)
        print(atom.get_positions())
        atom.get_scaled_positions()
    else:
        raise TypeError("Model should be Atoms")
    return(atom)

def cal_strain(ini_atoms, pre_atoms, isprint=False):
    '''
    函数用于计算应变，变形前模型：ini_atoms, 变形后模型：pre_atoms，两个模型的属性是 ase.atoms.Atoms，
    如果两个模型不是Atoms(Type Error)，或者不具有应变变换(Value Error)，会提示错误。
    '''

    isAtoms = isinstance(ini_atoms, Atoms) + isinstance(pre_atoms, Atoms)
    # len_queal = len(ini_atoms.positions) == len(pre_atoms.positions)
    if isAtoms != 2:
        print("Two model are Atoms:", isAtoms == 2)
        # print("Two models with equal atomic numbers :", len_queal == 1)
        raise TypeError("Model should be Atoms")
    ini_cor = ini_atoms.cell.array
    pre_cor = pre_atoms.cell.array
    # 计算形变梯度张量
    F = np.matmul(pre_cor, np.linalg.inv(ini_cor))
    E = 1 / 2 * (F.T + F) - np.identity(3)
    if isprint:
        print("strain: \n", E)
    return E

def opt_strain(bulk, strain, iscal=False):

    # 在bulk上施加应变
    strain_bulk = bulk.copy()
    strain_real = 0
    # 获取当前的晶格矩阵, 复制初始无应变构型
    cell = np.array(strain_bulk.get_cell())
    nostrain = np.identity(3)  # 单位矩阵
    # 在 x 方向上添加应变
    F = nostrain + np.array(
        [[strain[0], strain[2], 0], [strain[2], strain[1], 0], [0, 0, 0]]
    )
    newcell = np.matmul(F, cell)
    # 将新的晶格矩阵赋值给 原始 Cu 对象
    strain_bulk.set_cell(newcell, scale_atoms=True)
    # scale_Atoms=True must be set to True to ensure that ...
    # ...the atomic coordinates adapt to changes in the lattice matrix
    if iscal:
        strain_real = extracted_from_opt_strain(
            strain_bulk, lammps, cell, bulk
        )
    else:
        strain_real = cal_strain(bulk, strain_bulk)
    return (strain_bulk, strain_real)


def opt_strain_F(bulk, strain, iscal=False):

    # 在bulk上施加应变
    strain_bulk = bulk.copy()
    strain_real = 0
    # 获取当前的晶格矩阵, 复制初始无应变构型
    cell = np.array(strain_bulk.get_cell())
    nostrain = np.identity(3)  # 单位矩阵
    # 在 x 方向上添加应变
    F = nostrain + np.array(
        [[strain[0], 0, 0], [0, strain[1], 0], [0, 0, strain[2]]]
    )
    newcell = np.matmul(F, cell)
    # 将新的晶格矩阵赋值给 原始 Cu 对象
    strain_bulk.set_cell(newcell, scale_atoms=True)
    # scale_Atoms=True must be set to True to ensure that ...
    # ...the atomic coordinates adapt to changes in the lattice matrix
    if iscal:
        strain_real = extracted_from_opt_strain(
            strain_bulk, lammps, cell, bulk
        )
    else:
        strain_real = cal_strain(bulk, strain_bulk)
    return (strain_bulk, strain_real)

def extracted_from_opt_strain(strain_bulk, lammps, cell, bulk):
    # 施加应变后的模型，lammps可读文件
    write("strain.lmp", strain_bulk, format="lammps-data")  # type: ignore
    # 执行lammps优化，固定了x和y的自由度，只放松了z方向的自由度
    infile = "in.strain.in"
    lmp = lammps()
    lmp.file(infile)
    atoms = lammpsdata.read_lammps_data("opt_strain.data", style="atomic")

    new_cell = atoms.get_cell()
    dot_cell = np.dot(cell[0], cell[1])
    dot_new = np.dot(new_cell[0], new_cell[1])
    if dot_cell * dot_new < 0:  # 与基础的基矢量构型不同
        new_cell[1] = new_cell[1] + new_cell[0]

    strain_bulk.set_cell(new_cell, scale_atoms=True)
    return cal_strain(bulk, strain_bulk)

def cal_LBFGC(ini_model, potential=EMT, fmax=1e-6, steps=1e3):
    # 执行动力学过程，默认的势函数是EMT，力收敛判断值1E-6，最大动力学步数1E3
    # 这个优化似乎不能放缩盒子
    ini_model.set_calculator(potential())  # setting the calculated potential
    # 创建 LBFGS 实例
    dyn = LBFGS(ini_model, logfile="None")
    # 进行能量最小化优化计算
    dyn.run(fmax, steps)
    # 输出优化后的结构信息和能量值
    opt_config = dyn.atoms  # initial model
    opt_energy = dyn.atoms.get_potential_energy()
    return (opt_config, opt_energy)

def cal_BFGC(ini_model, potential=EMT, fmax=1e-6, steps=1000):
    # 执行动力学过程，默认的势函数是EMT，力收敛判断值1E-6，最大动力学步数1E3
    # 这个优化似乎不能放缩盒子
    ini_model.set_calculator(potential())  # setting the calculated potential
    # 创建 BFGS 实例
    dyn = BFGS(ini_model)
    # 进行能量最小化优化计算
    dyn.run(fmax, steps)

    # 输出优化后的结构信息和能量值
    opt_config = dyn.atoms  # initial model
    opt_energy = dyn.atoms.get_potential_energy()
    print("Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm")
    return (opt_config, opt_energy)

def set_strain(ini_model, strain=None, is_opt=True): # 该函数保留未使用
    if strain is None:
        strain = [0, 0, 0]
    # strain 表面应变由三个值控制 [ε1 ε2 ε6]
    isAtoms = isinstance(ini_model, Atoms)
    if isAtoms == 0:
        raise TypeError("Model should be Atoms")

    strain_slab = ini_model.copy()
    # 获取当前的晶格矩阵, 复制初始无应变构型
    cell = strain_slab.get_cell()

    strains = np.array([[strain[0], strain[2]], [strain[2], strain[1]]])
    deform = strains + np.identity(2)
    cell[:2, :2] = np.dot(cell[:2, :2], deform)
    # 将新的晶格矩阵赋值给 原始 Cu 对象
    strain_slab.set_cell(cell, scale_atoms=True, apply_constraint=False)
    # scale_Atoms=True must be set to True to ensure that ...
    # ...the atomic coordinates adapt to changes in the lattice matrix

    if is_opt == True:
        opt_strain_slab, opt_strain_energr = cal_LBFGC(strain_slab)
    else:
        opt_strain_slab, opt_strain_energr = strain_slab, False
    # strain = compare_atoms(ini_model,strain_slab)
    return (opt_strain_slab, opt_strain_energr)

def copy_contcar(
    rootdir,
    destdir=None,
    input_file="CONTCAR",
    output_put="POSCAR",
    func=shutil.copy2,
):
    # 该函数起到读取目录，构建对应目录，执行操作生成文件三个功能。
    # rootdir：已有的目录, destdir=None： 默认的生成的新目录, input_file='CONTCAR'：已有的文件,output_put='POSCAR'：生成的文件
    # func：源文件到新文件之间的操作，默认的操作是复制
    if destdir is None:
        destdir = f"{rootdir} _copy"
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    poscar_files = []
    for dirpath, _, filenames in os.walk(rootdir):
        # 构建对应的目标文件夹路径
        destpath = dirpath.replace(rootdir, destdir)
        if not os.path.exists(destpath):
            os.makedirs(destpath)
        for filename in filenames:
            if filename == input_file:
                # 构建新的文件名
                new_filename = output_put
                # 构建源文件路径和目标文件路径
                src_file = os.path.join(dirpath, filename)
                dest_file = os.path.join(destpath, new_filename)
                # 复制文件到目标路径
                func(src_file, dest_file)
                poscar_files.append(src_file)
                poscar_files = sorted(poscar_files)
    return poscar_files


def sort_z(z):
    tolerate = (max(z) - min(z)) / 1e3
    number = np.arange(len(z))  # 将 number 转换为 NumPy 数组
    adj = abs(z[:, None] - z[None, :]) < tolerate
    finish = []
    sort_list = []

    for i in range(len(z)):
        if i not in finish:
            # 找出与 z[i] 相似的所有索引
            similar_indices = number[adj[i]]  # 使用布尔索引
            sort_list.append(similar_indices)  # 添加索引列表到 sort_list
            finish.extend(similar_indices)  # 将索引逐一添加到 finish

    # sort_list 存储了所有符合条件的索引组合
    # finish 存储了所有处理过的索引
    # 对 sort_list 进行排序，依据每个子列表的平均值
    sort_list_order = sorted(sort_list, key=lambda x: (sum(z[x]) / len(x)) )

    s_z = np.zeros(z.shape)
    for i in number:
        indices = [index for index, sublist in enumerate(sort_list_order) if i in sublist]
        s_z[i] = indices[0]
    return(s_z)

def build_suface(
    bulk, vacuum_height=15.0, cell_z=None, relax_depth=2, iscala=False
):
    # 默认放松的原子层厚度为2
    if cell_z is not None:
        vacuum_height = cell_z - bulk.get_cell()[-1, -1]
        # 如果指定了期望的最终的cell的z方向高度，将采用期望高度
    # 创建真空层，生成表面
    adslab = bulk.copy()  # 无应变bulk构型
    if len(adslab.constraints):
        _extracted_from_build_suface_9(adslab, relax_depth)
    add_vacuum(adslab, vacuum_height)
    adslab.set_pbc(True)
    if iscala:
        adslab, adslab_e = cal_LBFGC(adslab)
    return adslab

def _extracted_from_build_suface_9(adslab, relax_depth):
    tags = np.zeros(len(adslab))
    layers = sort_z(adslab.get_positions()[:, 2])
    filtered_layers = [
        i for i in range(len(layers)) if layers[i] > max(layers) - relax_depth
    ]  #
    tags[filtered_layers] = 2  # 被吸附基底的表面层
    adslab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag < 1)])
    adslab.set_constraint(cons)

def vac_ext(atom, vacuum_h=0.0, ads_layer=0):
    # 在原本的bulk上补充多层原子，符合构造规律，然后在z轴增加真空层
    # 相对于只允许复制的方式来扩展晶胞z方向，该命令运行可以很方便地指定扩展的层数
    slab = atom.copy()
    slab2 = slab.copy()
    pos = slab.get_positions()
    layers1 = sort_z(pos[:, 2])
    maxl = 2 + math.ceil(ads_layer / (max(layers1) + 1))
    slab2 = slab2.repeat((1, 1, maxl))
    pos2 = slab2.get_positions()
    layers2 = sort_z(pos2[:, 2])

    filtered_layers = [
        i for i in range(len(layers2)) if layers2[i] > ads_layer + max(layers1)
    ]  # 标记高于需求层之上的部分
    del slab2[filtered_layers]  # 大于需求层数的删去

    slab = slab2.copy()
    slab = build_suface(slab, cell_z=vacuum_h, iscala=False)

    layers3 = sort_z(slab.get_positions()[:, 2])
    filtered_layers2 = [
        i for i in range(len(layers3)) if layers3[i] > max(layers3) - 2
    ]  #
    tags = np.zeros(len(slab))
    tags[filtered_layers2] = 2  # 被吸附基底的表面层
    slab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons0 = FixAtoms(indices=[atom.index for atom in slab if (atom.tag < 1)])
    cons1 = FixCartesian(filtered_layers2, mask=(1, 1, 0))
    slab.set_constraint([cons0, cons1])
    slab.set_pbc(True)
    # 打印更新后的结构
    return slab

def operate(src_file, dest_file):
    bulk = read(src_file)
    slab = vac_ext(bulk)
    write(dest_file, slab, format="vasp")


def visiual_atoms_strain(atoms, strain, isnormal=False, strain_range=None):
    # 创建一个 1x3 的子图布局，dpi 设置为 600
    fig, axs = plt.subplots(1, 3, dpi=600)

    # 调整子图之间的间距
    fig.subplots_adjust(wspace=0.4, hspace=0)

    # 绘制每个子图
    for i in range(3):
        ax = axs[i]  # 获取当前子图的坐标轴

        s = strain[:,i,i]
        if i==2:
            s = strain[:,0,1]
        cmap = plt.colormaps['RdYlBu'].reversed()

        # 归一化数据范围，使得数据的最小值对应于色图的最小值，最大值对应于色图的最大值
        if isnormal:
            norm = mcolors.Normalize(vmin=min(strain.reshape(-1)), vmax=max(strain.reshape(-1)))
        elif strain_range is None:
            vmin=min(s.reshape(-1))
            vmax=max(s.reshape(-1))

            if (vmax-vmin)< 1E-3:
                print('y')
                vmax = vmax+1E-3
                vmin = vmin-1E-3
            abs_max = max(abs(vmin), abs(vmax))
            strain_range = [-abs_max,abs_max]
        norm = mcolors.Normalize(vmin=strain_range[0], vmax=strain_range[1])
        # norm = mcolors.Normalize(vmin=min(s.reshape(-1)), vmax=max(s.reshape(-1)))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 空数组，表示没有实际数据用于 colorbar
        cbar = ax.figure.colorbar(sm, ax=axs[i], orientation='vertical',fraction=0.05, pad=0.05, shrink=0.25,)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))  # 转换为百分比格式
        cbar.ax.tick_params(labelsize=8)  # 设置字体大小为8，可以根据需要调整

        # 根据归一化的数值列表生成对应的颜色
        colors = [cmap(norm(value)) for value in s]

        Matplotlib(atoms, ax, colors=colors, radii=1.25*np.ones(len(atoms))).write()

        axs[i].set_xticks([])  # 关闭x轴的刻度
        axs[i].set_yticks([])  # 关闭y轴的刻度
        axs[i].set_xticklabels([])  # 关闭x轴的数字
        axs[i].set_yticklabels([])  # 关闭y轴的数字
    # 显示绘制的图形
        # if i==2 and isnormal:
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #     sm.set_array([])  # 空数组，表示没有实际数据用于 colorbar
        #     cbar = ax.figure.colorbar(sm, ax=axs[i], orientation='vertical',fraction=0.05, pad=0.05, shrink=0.25)
        #     cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*100}%'))  # 转换为百分比格式
    plt.show()


class strain_system2:
    def __init__(self, reference: Atoms, deformation: Atoms, ads_list=None):
        self.deformation = deformation
        self.reference = reference
        self.cutoffs = natural_cutoffs(self.reference)
        self.ads_list = ads_list
        nl = NeighborList(self.cutoffs, self_interaction=False, bothways=True)
        nl.update(self.reference)
        self.nl = nl

        cutoffs = natural_cutoffs(self.deformation)
        nl = NeighborList(np.array(cutoffs), self_interaction=False, bothways=True)
        nl.update(self.deformation)
        self.nl_deformation = nl

    def get_vector(self, num):
        a = self.reference
        b = self.deformation
        nl_num = self.nl.get_neighbors(num)
        # if num == 29:
        #     print(nl_num)
        reference_vectors = np.array([a.get_distance(num, i, mic=True, vector=True) for i in nl_num[0]])
        deformation_vectors = np.array([b.get_distance(num, i, mic=True, vector=True) for i in nl_num[0]])
        return(reference_vectors, deformation_vectors)

    def falk_x(self, reference_vectors, deformation_vectors):
        fx = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                fx[i, j] = sum(v1[i] * v2[j] for v1, v2 in zip(reference_vectors, deformation_vectors))
        return fx

    def falk_y(self, refermation_vectors):
        fy = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                fy[i, j] = sum(v[i] * v[j] for v in refermation_vectors)
        return fy

    def affine_deformation(self, fx, fy):
        y_inv = np.linalg.inv(fy)
        deform = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                deform[i, j] = sum(fx[i, k] * y_inv[k, j] for k in range(3))
        deform = deform - np.eye(3)
        return deform

    def get_strain(self):
        strain_filed = np.zeros((len(self.reference), 3, 3))
        ref_list = self.ads_list
        revers_list = None
        if ref_list is None:
            ref_list = range(len(self.deformation))
        else:
            ref_list0 = range(len(self.deformation))
            revers_list = [ref_list0[r] for r in ref_list]
            ref_list = list(set(ref_list0) - set(revers_list))
        for i in ref_list:
            rv, dv = self.get_vector(i)
            fx = self.falk_x(rv, dv)
            fy = self.falk_y(rv)
            self.affine_deformation(fx,fy)
            strain_i = self.affine_deformation(fx, fy)
            strain_filed[i] = strain_i
        if revers_list is not None:
            zlayers = sort_z(self.reference.get_positions()[:,-1])
            pos = self.deformation.get_positions()
            ads_center = np.mean(pos[revers_list,:], axis=0)
            connect_atoms = []
            for ads_atoms in revers_list:
                ca = self.nl_deformation.get_neighbors(ads_atoms)[0]
                connect_atoms.extend(list(ca))
            for first_atom in connect_atoms:
                sa = self.nl.get_neighbors(first_atom)[0]
                sa = list(set(sa)-set(connect_atoms))
                same_layer = np.where(zlayers == zlayers[first_atom])[0]
                sa = list(set(sa) & set(same_layer))
                length = np.linalg.norm(pos[first_atom,:]-ads_center)
                refer_length = np.linalg.norm(pos[sa,:]-ads_center, axis=1)
                ratio = (refer_length/length)**2
                refer_stress = np.array(strain_filed)[sa,:,:] * ratio[:, np.newaxis, np.newaxis]
                refer_stress = np.mean(refer_stress, axis=0)
                strain_filed[first_atom] = refer_stress
        self.strain_filed = strain_filed

        return(self.strain_filed)


class strain_system:
    #
    def __init__(self, reference: Atoms, deformation: Atoms, ads_list=None):
        self.deformation = deformation
        self.reference = reference
        self.cutoffs = np.array(natural_cutoffs(self.reference))*1.2
        self.ads_list = ads_list
        self.revers_list = None
        self.strain_filed = np.zeros((len(self.reference), 3, 3))
        nl = NeighborList(self.cutoffs, self_interaction=False, bothways=True)
        nl.update(self.reference)
        self.nl = nl
        # 关键变量，紧邻关系
        cutoffs = np.array(natural_cutoffs(self.deformation))*1.2
        nl_d = NeighborList(np.array(cutoffs), self_interaction=False, bothways=True)
        nl_d.update(self.deformation)
        self.nl_deformation = nl_d

    def get_vector(self, num):
        a = self.reference
        b = self.deformation
        nl_num,_ = self.nl.get_neighbors(num)
        # print(nl_num)
        if self.revers_list is not None:
            nl_num = np.setdiff1d(nl_num, self.revers_list, assume_unique=True)
        reference_vectors = np.array([a.get_distance(num, i, mic=True, vector=True) for i in nl_num])
        deformation_vectors = np.array([b.get_distance(num, i, mic=True, vector=True) for i in nl_num])
        # print(reference_vectors)
        # print(deformation_vectors)

        return(reference_vectors, deformation_vectors)

    def falk_x(self, reference_vectors, deformation_vectors):
        fx = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                fx[i, j] = sum(v1[i] * v2[j] for v1, v2 in zip(reference_vectors, deformation_vectors))
        return fx

    def falk_y(self, refermation_vectors):
        fy = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                fy[i, j] = sum(v[i] * v[j] for v in refermation_vectors)
        return fy

    def affine_deformation(self, fx, fy):
        y_inv = np.linalg.pinv(fy)
        deform = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                deform[i, j] = sum(fx[i, k] * y_inv[k, j] for k in range(3))
        deform = deform - np.eye(3)
        return deform

    def get_strain(self):
        strain_filed = np.zeros((len(self.reference), 3, 3))
        if self.ads_list is None:
            ref_list = range(len(self.deformation))
            revers_list = None
        else:
            ref_list0 = range(len(self.deformation))
            ads_list = [ref_list0[r] for r in self.ads_list] # 序号转正
            ref_list = list(set(ref_list0) - set(ads_list))
            # print(ads_list)
            revers_list = [self.nl_deformation.get_neighbors(i)[0] for i in ads_list]
            revers_list = np.concatenate(revers_list)
            # ref_list中是去除掉吸附或间隙原子的列表，revers_list是吸附或间隙原子
        self.revers_list = revers_list
        # print(revers_list)

        for i in ref_list:
            rv, dv = self.get_vector(i)
            fx = self.falk_x(rv, dv)
            fy = self.falk_y(rv)
            self.affine_deformation(fx,fy)
            strain_i = self.affine_deformation(fx, fy)
            strain_filed[i] = strain_i

        self.strain_filed = strain_filed

        return(self.strain_filed)


from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def visiual_atoms_strain_3d(atoms, strain, isnormal=False, input_range=None, type='strain', units=None):
    plt.rcParams.update({'font.size': 10})

    # 主网格布局：1行4列，最后一列单独处理
    fig = plt.figure(dpi=200, figsize=(9, 3))
    main_gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.2)  # 前三列等宽，第四列窄

    # 前三列放置原子图
    ax1 = fig.add_subplot(main_gs[0, 0])
    ax2 = fig.add_subplot(main_gs[0, 1])
    ax3 = fig.add_subplot(main_gs[0, 2])
    axs = [ax1, ax2, ax3]

    # 第四列内部嵌套3行网格（上留白、颜色条、下留白）
    inner_gs = GridSpecFromSubplotSpec(3, 1, height_ratios=[1, 3, 1], subplot_spec=main_gs[0, 3], hspace=0)
    cax = fig.add_subplot(inner_gs[1, 0])  # 中间行用于颜色条

    strain_symbol = ['x', 'y', 'z']

    pos = atoms.get_positions()
    sz = sort_z(pos[:,-1])
    atoms_z = atoms[sz>max(sz)-3]

    for i in range(3):
        ax = axs[i]
        s = strain[:, i, i]
        cmap = plt.colormaps['RdYlBu'].reversed()

        # 归一化逻辑
        if isnormal:
            norm = mcolors.Normalize(vmin=strain.min(), vmax=strain.max())
        else:
            if input_range is None:
                vmin = s.min()
                vmax = s.max()
                if (vmax - vmin) < 1e-3:
                    vmax += 1e-3
                    vmin -= 1e-3
                abs_max = max(abs(vmin), abs(vmax))
                input_range = [-abs_max, abs_max]
            norm = mcolors.Normalize(vmin=input_range[0], vmax=input_range[1])

        colors = np.array([cmap(norm(value)) for value in s])
        colors_sz = colors[sz>max(sz)-3]

        Matplotlib(atoms_z, ax, colors=colors_sz).write() # 仅仅画了垂直于方向的三层原子

        # Matplotlib(atoms_z, ax, colors=colors_sz).write()

        # 设置标题
        subscript = f'{strain_symbol[i]}{strain_symbol[i]}'
        title = r'$\varepsilon _{' + subscript + '}$' if type == 'strain' else r'$\sigma _{' + subscript + '}$'
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()

    # 仅在最后一个循环外添加颜色条（避免重复）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if type == 'strain':
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', aspect=15)
        cbar.set_label(r'Strain (%)', size=12)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*100:.1f}'))
    elif type == 'stress':
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', aspect=15)
        if units is None:
            symbel_units = 'Stress (GPa)'
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
        else:
            symbel_units = f'Stress ({units})'
            # cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
        cbar.set_label(symbel_units, size=12)


    cbar.ax.tick_params(labelsize=10)

    plt.show()


def visiual_atoms_strain_tensor(atoms, strain, isnormal=False, strain_range=None):
    # 创建一个 1x3 的子图布局，dpi 设置为 600
    fig, axs = plt.subplots(3, 3, dpi=600)
    # 调整子图之间的间距
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    strain_symbol = ['x','y','z']
    # 绘制每个子图
    for i in range(3):
        for j in range(3):
            ax = axs[i][j]  # 获取当前子图的坐标轴

            s = strain[:,i,j]
            # if i==2:
            #     s = strain[:,0,1]
            cmap = plt.colormaps['RdYlBu'].reversed()

            # 归一化数据范围，使得数据的最小值对应于色图的最小值，最大值对应于色图的最大值
            if isnormal:
                norm = mcolors.Normalize(vmin=min(strain.reshape(-1)), vmax=max(strain.reshape(-1)))
            elif strain_range is None:
                vmin=min(s.reshape(-1))
                vmax=max(s.reshape(-1))
                # print(vmin, vmax)
                if (vmax-vmin)< 1E-3:
                    # print('y')
                    vmax = vmax+1E-3
                    vmin = vmin-1E-3
                abs_max = max(abs(vmin), abs(vmax))
                strain_range = [-abs_max,abs_max]
                # print(strain_range)
                norm = mcolors.Normalize(vmin=strain_range[0], vmax=strain_range[1])
            elif strain_range is not None:
                norm = mcolors.Normalize(vmin=strain_range[0], vmax=strain_range[1])

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 空数组，表示没有实际数据用于 colorbar
            cbar = ax.figure.colorbar(sm, ax=axs[i][j], orientation='vertical',fraction=0.12, pad=0.05, shrink=0.6,)
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*100:.2f}%'))  # 转换为百分比格式
            cbar.ax.tick_params(labelsize=8)  # 设置字体大小为8，可以根据需要调整

            # 根据归一化的数值列表生成对应的颜色
            colors = [cmap(norm(value)) for value in s]

            Matplotlib(atoms, ax, colors=colors, radii=1.25*np.ones(len(atoms))).write()
            subscript = f'{strain_symbol[i]}{strain_symbol[j]}'
            axs[i][j].set_title(r'$\varepsilon _{' + subscript + '}$')
            axs[i][j].set_xticks([])  # 关闭x轴的刻度
            axs[i][j].set_yticks([])  # 关闭y轴的刻度
            axs[i][j].set_xticklabels([])  # 关闭x轴的数字
            axs[i][j].set_yticklabels([])  # 关闭y轴的数字

    plt.show()


def sort_lammps_data(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    header = []
    atoms = []
    in_atoms_section = False

    for line in lines:
        if line.startswith('Atoms'):
            header.append(line)
            in_atoms_section = True
            continue
        elif line.startswith('Velocities'):
            in_atoms_section = False
            break
        if in_atoms_section:
            if line.strip() == ' ':  # 处理空行作为原子部分结束标志
                in_atoms_section = False
                # header.append(line)
            else:
                # 去除行尾换行符
                atoms.append(line.rstrip('\n'))
        else:
            header.append(line.rstrip('\n'))  # 保持头部换行符一致
    # header.append(atoms[0])
    atoms = [s for s in atoms if s != ""]
    # print(atoms)
    # 按原子ID排序
    atoms.sort(key=lambda x: int(x.split()[0]))

    with open(output_file, 'w') as f:
        # 写入头部（保留原有换行符）
        f.write('\n'.join(header) + '\n')
        # 写入原子数据（每行仅一个换行符）
        f.write('\n'.join(atoms) + '\n')


from math import sin, cos
def get_rotation_matrix(theta, axis):
    """ 生成绕指定轴旋转的3D旋转矩阵 """
    theta = np.deg2rad(theta)
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)]
        ])
    elif axis == 'y':
        return np.array([
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)]
        ])
    elif axis == 'z':
        return np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x', 'y' or 'z'")
def plot_stress( model, forces, rotation="0x,0y,0z"):
    """ 绘制原子位置和力矢量 """
    # 解析 rotation 参数
    # rotation_str = "0x,0y,0z"  # 必须与 plot_atoms 使用的参数一致
    R_total = np.eye(3)
    for part in rotation.split(','):
        part = part.strip()
        angle_str = part[:-1]
        axis = part[-1].lower()
        angle = float(angle_str)
        R = get_rotation_matrix(angle, axis)
        R_total = R @ R_total  # 累积旋转

    # 处理原子位置和力矢量
    positions = model.get_positions()
    rot_positions = positions @ R_total.T  # 应用旋转
    x, y = rot_positions[:, 0], rot_positions[:, 1]

    rot_forces = forces @ R_total.T  # 应用相同旋转
    fx, fy = rot_forces[:, 0], rot_forces[:, 1]
    # 创建一个新的图形和坐标轴
    fig, axs = plt.figure()
    # fig, axs = plt.subplots()

    m = Matplotlib(model, axs, radii=0.9, rotation=(rotation))
    offset = m.cell_vertices[0]
    x += offset[0]
    y += offset[1]
    m.write()
    scale = 0.1  # 根据力的大小调整这个值
    axs.quiver(x, y, fx, fy,
            color='r', scale=1/scale,
            width=0.005, headwidth=3,
            headlength=4, zorder=10)

    plt.show()

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import h5py

def save_data_to_hdf5(filename, bulk_md, field, type='strain'):

    # 原子坐标和对应值
    pos = bulk_md.get_positions()
    pos2 = pos.copy()
    cell = bulk_md.get_cell()
    cell_z = cell[-1][-1]/2
    pos[pos2[:,-1]<cell_z, -1] = pos2[pos2[:,-1]<cell_z, -1]+cell_z
    pos[pos2[:,-1]>cell_z, -1] = pos2[pos2[:,-1]>cell_z, -1]-cell_z
    # value = strain_field_is[:,0,0]
    num_levels = 9
    vmin, vmax = -0.002,  0.002
    values_range = np.linspace(vmin, vmax, num_levels)

    # 检查数据范围
    # print(f"原始数据范围: [{np.min(value):.2e}, {np.max(value):.2e}]")
    # print(f"等值面层级设置: {values_range}")

    # 高分辨率插值设置 -------------------------------------------------
    num_points = 100  # 平衡精度和性能
    x = np.linspace(pos[:,0].min(), pos[:,0].max(), num_points)
    y = np.linspace(pos[:,1].min(), pos[:,1].max(), num_points)
    cell_z = np.linspace(pos[:,2].min(), pos[:,2].max(), num_points)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, cell_z, indexing='ij')

    # 创建HDF5文件
    with h5py.File(filename, 'w') as f:
        group = f.create_group('/000000')

        # 写入坐标数据
        group.create_dataset('x', data=grid_x)
        group.create_dataset('y', data=grid_y)
        group.create_dataset('z', data=grid_z)
        grid_strian = {}
        for i in range(3):
            for j in range(i+1):
                if type == 'strain':
                    key = f'e{j+1}{i+1}'
                elif type == 'stress':
                    key = f's{j+1}{i+1}'
                else:
                    raise ValueError("Invalid type. Use 'strain' or 'stress'.")
                value = field[:,j,i]
                grid_values = griddata(
                    pos,
                    value,
                    (grid_x, grid_y, grid_z),
                    method='linear',
                    fill_value=np.nan
                )
                grid_values = np.nan_to_num(grid_values, nan=np.mean(value))  # 均值填充
                grid_values = gaussian_filter(grid_values, sigma=2)  # 适当平滑
                grid_strian[key] = grid_values
        # 写入应变分量
        if type == 'strain':
            for component in ['e11', 'e12', 'e13', 'e22', 'e23', 'e33']:
                group.create_dataset(component, data=grid_strian[component])
        elif type == 'stress':
            for component in ['s11', 's12', 's13', 's22', 's23', 's33']:
                group.create_dataset(component, data=grid_strian[component])
        else:
            raise ValueError("Invalid type. Use 'strain' or 'stress'.")


def plot_stress( model, forces, rotation="0x,0y,0z"):
    """ 绘制原子位置和力矢量 """
    # 解析 rotation 参数
    # rotation_str = "0x,0y,0z"  # 必须与 plot_atoms 使用的参数一致
    R_total = np.eye(3)
    for part in rotation.split(','):
        part = part.strip()
        angle_str = part[:-1]
        axis = part[-1].lower()
        angle = float(angle_str)
        R = get_rotation_matrix(angle, axis)
        R_total = R @ R_total  # 累积旋转

    # 处理原子位置和力矢量
    positions = model.get_positions()
    rot_positions = positions @ R_total.T  # 应用旋转
    x, y = rot_positions[:, 0], rot_positions[:, 1]

    rot_forces = forces @ R_total.T  # 应用相同旋转
    fx, fy = rot_forces[:, 0], rot_forces[:, 1]
    # 创建一个新的图形和坐标轴
    fig = plt.figure()
    axs = fig.add_subplot(111)  # 添加一个 1x1 的网格中的第1个坐标系

    m = Matplotlib(model, axs, radii=0.9, rotation=(rotation))
    offset = m.cell_vertices[0]
    x += offset[0]
    y += offset[1]
    m.write()
    scale = 0.1  # 根据力的大小调整这个值
    axs.quiver(x, y, fx, fy,
            color='r', scale=1/scale,
            width=0.005, headwidth=3,
            headlength=4, zorder=10)

    plt.show()

def covert6to3d(is_force):
    """
    将6个分量转换为3x3矩阵
    :param is_force: 6个分量
    :return: 3x3矩阵
    """
    # 创建一个3x3的零矩阵
    is_stress_md = np.zeros((is_force.shape[0], 3, 3))

    # 填充对角线元素
    is_stress_md[:, 0, 0] = is_force[:, 0]  # xx
    is_stress_md[:, 1, 1] = is_force[:, 1]  # yy
    is_stress_md[:, 2, 2] = is_force[:, 2]  # zz

    # 填充非对角线元素（对称位置）
    is_stress_md[:, 0, 1] = is_stress_md[:, 1, 0] = is_force[:, 3]  # xy
    is_stress_md[:, 0, 2] = is_stress_md[:, 2, 0] = is_force[:, 4]  # xz
    is_stress_md[:, 1, 2] = is_stress_md[:, 2, 1] = is_force[:, 5]  # yz

    return is_stress_md


def extract_atoms_data(filename, order=-1):
    # 读取文件并预处理每一行
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]

    # 分割数据块
    blocks = []
    current_block = []
    for line in lines:
        if line.startswith("ITEM: TIMESTEP"):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)

    # 提取每个块中的原子数据
    all_data = []
    for block in blocks:
        data = []
        in_atoms_section = False
        for line in block:
            if line.startswith("ITEM: ATOMS"):
                in_atoms_section = True
                continue
            if in_atoms_section:
                data.append(line.split())
        if data:
            all_data.append(data)
    stress = all_data[order] # 提取指定块的原子数据, 默认是最后一个块
    stress_array = np.array([[float(elem) for elem in sublist] for sublist in stress])
    # 将数据转换为NumPy数组
    return stress_array