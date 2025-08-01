#!/bin/bash
# designed by zwbai

# 使用glob扩展获取文件列表（避免通配符扩展失败）
shopt -s nullglob  # 如果没有匹配到文件，避免保留通配符
files=(in.*.in CHONiPt.ff)  # 这里直接用数组获取文件

# 获取当前目录下的所有子文件夹（安全处理空格）
find . -mindepth 1 -type d | while IFS= read -r folder; do
    # 进入子文件夹
    pushd "$folder" > /dev/null
    
    # 获取当前文件夹名
    current_folder="${PWD##*/}"
    echo "current folder name: $current_folder"
    
    # 复制文件到当前子文件夹
    for file in "${files[@]}"; do
        if [[ -e ../$file ]]; then
            echo "Copy $file to $folder"
            cp "../$file" .
        else
            echo "Warning: file $file absent"
        fi
    done
    
    # 返回上级目录
    popd > /dev/null
done