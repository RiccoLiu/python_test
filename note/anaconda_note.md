# Anaconda 笔记

## Anaconda 环境搭建

### Ubuntu 系统

下载安装包：  
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh  

安装安装包：  
./Anaconda3-2024.10-1-Linux-x86_64.sh

初始化：  
~/anaconda3/bin/conda init

重新加载配置：  
source ~/.bashrc

### Conda 基础命令
- 查看版本  
conda --version  

- 查看信息  
conda info  

- 创建新环境  
conda create --name myenv python=3.10  

- 激活环境  
conda activate myenv  

- 安装软件包  
conda install numpy pandas  

- 退出环境  
conda deactivate  

- 查看虚拟环境列表  
conda env list  

- 删除环境  
conda remove -n py_3.10 --all  

- 更新conda  
conda update conda  
conda update anaconda  

- 编辑conda 配置文件  
conda config --set show_channel_urls yes  

- 添加镜像源  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  

- 更新缓存  
conda clean -i  

- 导出环境  
pip freeze > pip_requirements.txt  
conda list -e > conda_requirements.txt  
conda env export --name py3.10 > environment.yml  
conda env export --name py3.10 --no-builds > py3.10.yml  

- 导入环境  
conda env create -f environment.yml  

- 更新环境
conda env update -n base -f conda_environment.yaml  

- 禁用自动激活  
conda config --set auto_activate_base false  

