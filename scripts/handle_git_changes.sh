#!/bin/bash

# 处理本地git修改的脚本

echo "处理本地修改..."
git add .
git commit -m "临时提交: 更新模型配置和下载脚本"
git pull

echo "操作完成"
