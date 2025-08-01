#!/bin/bash
# 该脚本的目的是能够将该文件夹内的计算依次执行
# 设置传入参数的默认值
work_node=""
script_name=""

while getopts ":n:s:" opt; do
    case $opt in
        n) work_node="$OPTARG"
        ;;
        s) script_name="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done
if [ -z $work_node ]; then
    echo "no default node"
else
    echo "The work node is: $work_node"
fi
if [ -z $script_name ]; then
    echo "no default script"
else
    echo "The script is: $script_name"
fi

current_dir=$(pwd)
echo "current dir $current_dir"

# 获取当前文件夹下的子文件夹列表，并按 ASCII 码排序，过滤掉目录（.）
subfolders=( $(find . -maxdepth 1 -type d | sort | sed '/^\.$/d') )
length=${#subfolders[@]}
for ((i=0; i<length; i++))
do
    cd ${subfolders[i]}
    echo "cycle $((i)) calculate: ${subfolders[i]}"
    if [ -z $script_name ]; then
        echo "no extra script"
    else
        cp ../${script_name} ./
        chmod 777 ./${script_name}
        ./${script_name} >>out &
    fi
    # run
    core=$(cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $4}')
    echo "The work node parameter is: $work_node"
    
    mpiexec -np 8 lmp -in *.in -pk omp 2 -sf omp
    
    mpirun_pid=$!
    wait $mpirun_pid
    cd $OLDPWD
done
echo 'success'
