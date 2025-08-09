#!/bin/bash
#SBATCH -J vasp
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -p xahcnormal

module purge
module load compiler/intel/2017.5.239
module load mpi/intelmpi/2017.4.239

export MKL_DEBUG_CPU_TYPE=5 #加速代码
export MKL_CBWR=AVX2 #使cpu默认支持avx2
export I_MPI_PIN_DOMAIN=numa #内存位置与cpu位置绑定，加速内存读取。对于内存带宽要求高的计算提速明显

#srun --mpi=pmi2  /path/to/vasp_std #vasp程序路径
srun --mpi=pmi2 /work/home/acq6mir6nu/zwbai/software/home/zwbai/vasp.5.4.4-wenshan/bin/vasp_std
