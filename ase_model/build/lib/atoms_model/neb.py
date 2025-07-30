import os
import shutil
import numpy as np
from pathlib import Path
from os.path import join
import matplotlib.pyplot as plt
from ase.geometry import get_distances
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from .other import creat_folder, exist_folder
from deal_car import read_cars

import ase
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.utils.forcecurve import fit_images

def get_displacement(atoms1, atoms2):
    # 计算两个构型的距离
    cell = atoms1.cell
    pbc = atoms1.pbc
    _, D_len = get_distances(atoms1.positions, atoms2.positions, cell=cell, pbc=pbc)
    diagonal_vectors = D_len.diagonal()
    # print(diagonal_vectors)
    magnitudes = np.linalg.norm(diagonal_vectors, axis=0)

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

def interpolate_plot(x, y, is_plot=True):
    # 三次样条 结合 三次埃尔米特样条插值器，迭代插值以获得参考点的斜率，并保证初末和过渡态的导数为0

    # 创建三次样条插值器（自动计算导数）
    # bc_type 设置边界条件：'clamped' 表示两端一阶导数为0
    cubic_spline = CubicSpline(x, y, bc_type='clamped')
    # 计算参考点处的斜率（一阶导数）
    reference_points = x.copy()
    slopes = cubic_spline.derivative()(reference_points)
    max_index = np.argmax(y)
    slopes[max_index] = 0.0
    # 创建三次埃尔米特样条插值器
    hermite_spline = CubicHermiteSpline(x, y, slopes)
    # 生成插值结果
    x_interp = np.linspace(x[0], x[-1], 100)
    y_interp = hermite_spline(x_interp)
    if is_plot:
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='original')
        plt.plot(x_interp, y_interp, label='interpolation')
        plt.show()

    return(x_interp, y_interp)

def copy_files_skip_existing(src_dir, dst_dir):
    """
    递归复制文件夹内容，自动跳过已存在的文件
    （保留目录结构，不覆盖目标文件）

    参数:
        src_dir (str): 源目录路径
        dst_dir (str): 目标目录路径
    """
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    # 遍历源目录（包含子目录）
    for root, _, files in os.walk(src_path):
        # 计算相对路径
        relative_path = Path(root).relative_to(src_path)
        dest_dir = dst_path / relative_path

        # 创建目标目录（如果不存在）
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            # 跳过已存在的文件
            if not dest_file.exists():
                try:
                    shutil.copy2(src_file, dest_file)
                    # print(f"已复制: {src_file} -> {dest_file}")
                except PermissionError:
                    print(f"权限不足无法复制: {src_file}")
                except Exception as e:
                    print(f"复制失败: {src_file} - 错误: {str(e)}")
            else:
                print(f"Ignore file: {dest_file}")

def creat_neb(is_fs_path, neb_name, neb_path, n_mage_prepared=3):
# is_fs_path 是存储初态和模态的文件夹，结构优化的结果

    n_mage_prepared = [n_mage_prepared] * len(neb_name) if isinstance(n_mage_prepared, int) else n_mage_prepared
    creat_folder(neb_path, is_delete=True)
    for i in range(len(neb_name)):
        neb_name_i = neb_name[i] # two bulk name
        neb_name_i2 = f"{neb_name_i[0]}-{neb_name_i[1]}" # diffusion name
        is_path = join(is_fs_path,neb_name_i[0])
        fs_path = join(is_fs_path,neb_name_i[1])
        is_bulk = join(is_path,'CONTCAR')
        fs_bulk = join(fs_path,'CONTCAR')
        neb_i_path = join(neb_path, neb_name_i2)

        os.system(f"dist.pl {is_bulk} {fs_bulk}")
        exist_folder(neb_i_path, is_creat=True)

        os.system(f"cd {neb_i_path} && nebmake.pl {is_bulk} {fs_bulk} {n_mage_prepared[i]}")

        is_path_0 = join(neb_i_path, '00')
        fs_path_1 = join(neb_i_path, f"0{n_mage_prepared[i]+1}")
        copy_files_skip_existing(is_path, is_path_0)
        copy_files_skip_existing(fs_path, fs_path_1)
    neb = read_cars(neb_path, car='POSCAR')
    return(neb)


class NebProcess:
    def __init__(self, reactant, product, neb_system='neb',
                 n_images=5, neb_fmax=0.1, cineb_fmax=0.05, steps=500,
                 savepath='./neb_results', calculator=EMT(), transition_state=None):
        """
        Initialize NEB calculation process

        Args:
            reactant (Atoms): Initial state structure
            product (Atoms): Final state structure
            neb_system (str): System name for identification
            n_images (int): Number of images in the band
            neb_fmax (float): Force convergence for initial NEB
            cineb_fmax (float): Force convergence for CI-NEB
            steps (int): Maximum optimization steps
            savepath (str): Path for saving results
            calculator (Calculator): ASE calculator to use
            transition_state (Atoms/str): TS structure or path to file
        """
        self.reactant = reactant.copy()
        self.product = product.copy()
        self.transition_state = transition_state
        self.neb_system = neb_system
        self.n_images = n_images
        self.neb_fmax = neb_fmax
        self.cineb_fmax = cineb_fmax
        self.steps = steps
        self.savepath = savepath
        self.calculator = calculator

        # Will be populated during calculation
        self.atom_configs = []
        self.neb = None
        self.fmax_history = []
        self.converged = False

    def run(self):
        """Execute the full NEB calculation workflow"""
        self._prepare_images()
        self._relax_endpoints()
        self._interpolate_band()
        self._run_neb()
        self._run_cineb()
        return self.converged

    def _prepare_images(self):
        """Create initial image configurations"""
        self.atom_configs = [self.reactant.copy() for _ in range(self.n_images - 1)]
        self.atom_configs.append(self.product.copy())

        # Set calculator for all images
        for atoms in self.atom_configs:
            atoms.calc = EMT()

    def _relax_endpoints(self):
        """Relax reactant and product structures"""
        print("Relaxing endpoints ...")
        BFGS(self.atom_configs[0]).run(fmax=self.neb_fmax)
        BFGS(self.atom_configs[-1]).run(fmax=self.neb_fmax)

    def _interpolate_band(self):
        """Create initial path with optional transition state"""
        print("Interpolating band ...")

        if self.transition_state:
            # Handle TS input (either Atoms object or file path)
            ts = read(self.transition_state) if isinstance(self.transition_state, str) else self.transition_state
            middle_idx = len(self.atom_configs) // 2
            self.atom_configs[middle_idx].set_positions(ts.get_positions())

            # Interpolate each segment separately
            first_band = NEB(self.atom_configs[: middle_idx + 1])
            second_band = NEB(self.atom_configs[middle_idx:])
            first_band.interpolate("idpp")
            second_band.interpolate("idpp")
        else:
            band = NEB(self.atom_configs)
            band.interpolate("idpp")

    def _run_neb(self):
        """Run initial NEB calculation"""
        print("Running NEB ...")
        self.neb = NEB(self.atom_configs, climb=False, parallel=True)
        self._run_optimization(self.neb_fmax)

    def _run_cineb(self):
        """Run CI-NEB calculation"""
        if not self.converged:
            print("NEB converged, running CI-NEB ...")
            self.neb.climb = True
            self.converged = self._run_optimization(self.cineb_fmax)

    def _run_optimization(self, fmax):
        """Run optimization with attached callbacks"""
        neb_tools = NEBTools(self.atom_configs)
        optimizer = NEBOptimizer(self.neb)

        # Attach monitoring functions
        optimizer.attach(self._check_calculations)
        optimizer.attach(self._write_to_db)
        optimizer.attach(lambda: self.fmax_history.append(neb_tools.get_fmax()))

        return optimizer.run(fmax=fmax, steps=self.steps)

    def _check_calculations(self):
        """Verify all images have calculation results"""
        for i, image in enumerate(self.atom_configs[1:-1]):
            if {"forces", "energy"} - set(image.calc.results.keys()):
                raise RuntimeError(f"Missing calculation for image {i+1}")

    def _write_to_db(self):
        """Write current results to database"""
        db_path = os.path.join(self.savepath, f'{self.neb_system}.db')
        with ase.db.connect(db_path) as db:
            for atoms in self.atom_configs:
                if atoms.calc.results:
                    db.write(atoms, data=atoms.calc.results)

    def plot_mep(self):
        """Plot the minimum energy path"""
        x = get_dist_list(self.neb.images)
        e = np.array(self.neb.energies)
        e = e - e[0]
        x_interp, y_interp = interpolate_plot(x, e)

        fig, ax = plt.subplots()
        ax.plot(x_interp, y_interp,
                label=f"Barrier: {max(y_interp):.2f} eV")

        # Formatting
        ax.patch.set_facecolor('#E8E8E8')
        ax.grid(color='w')
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel('Energy [eV]')
        ax.set_xlabel('Reaction Coordinate [Å]')
        ax.legend()

        return fig

    @property
    def barrier(self):
        """Get the reaction barrier in eV"""
        return max(NEBTools(self.atom_configs).get_fit().fit_energies)

    @property
    def transition(self):
        """Get the transition state image"""
        energies = [image.get_potential_energy() for image in self.atom_configs]
        return self.atom_configs[energies.index(max(energies))]

    @property
    def path_energies(self):
        """Get energies along the reaction path"""
        return NEBTools(self.atom_configs).get_fit().fit_energies

    @property
    def path_coordinates(self):
        """Get reaction coordinates"""
        return NEBTools(self.atom_configs).get_fit().fit_path


def plot_mep(images):
    fit = fit_images(images)

    fig, ax = plt.subplots()
    ax.plot(
        fit.fit_path, fit.fit_energies, label=f"Barrier: {max(fit.fit_energies):.2f} eV"
    )

    ax.patch.set_facecolor("#E8E8E8")
    ax.grid(color="w")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Energy [eV]")
    ax.set_xlabel("Reaction Coordinate [Å]")
    ax.legend()

    return fig

class CalculationChecker:
    def __init__(self, neb):
        self.neb = neb

    def check_calculations(self):
        missing_calculations = []
        for i, image in enumerate(self.neb.images[1:-1]):
            if {"forces", "energy"} - image.calc.results.keys():
                missing_calculations.append(i)

        if missing_calculations:
            raise ValueError(f"missing calculation for image(s) {missing_calculations}")

class DBWriter:
    def __init__(self, db_path, atomss):
        self.atomss = atomss
        self.db_path = db_path

    def write(self):
        with ase.db.connect(self.db_path) as db:
            for atoms in self.atomss:
                if atoms.calc.results:
                    db.write(atoms, data=atoms.calc.results)

def interpolate_band(atom_configs, transition_state=None):
    if transition_state:
        transition_state = read(transition_state)
        ts_positions = transition_state.get_positions()
        middle_idx = len(atom_configs) // 2
        atom_configs[middle_idx].set_positions(ts_positions)
        first_band = NEB(atom_configs[: middle_idx + 1])
        second_band = NEB(atom_configs[middle_idx:])
        first_band.interpolate("idpp")
        second_band.interpolate("idpp")
    else:
        band = NEB(atom_configs)
        band.interpolate("idpp")
    return atom_configs
