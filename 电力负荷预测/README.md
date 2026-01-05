\# 电力负荷预测系统



\## 项目简介

基于深度学习的多变量电力负荷预测平台，支持ETTh2数据集分析和实时交互预测。



\## 功能特点

\- 📊 多变量数据探索分析

\- 🔬 特征相关性可视化

\- 🤖 TransformerRNNPlus深度学习模型

\- 🔮 实时负荷预测

\- 📈 多模型性能对比



\## 在线访问

访问地址：https://你的应用名.streamlit.app



\## 本地运行

```bash

pip install -r requirements.txt

streamlit run app.py

```

技术栈

前端: Streamlit



可视化: Plotly



数据处理: Pandas, NumPy



深度学习: TransformerRNNPlus



作者

zorovolker



text



\### \*\*1.4 安装Git并推送代码\*\*



\#### \*\*Windows安装Git\*\*

1\. 下载：https://git-scm.com/download/win

2\. 双击安装，全部默认设置

3\. 安装完成后，在项目文件夹右键 → "Git Bash Here"



\#### \*\*推送代码到GitHub\*\*

在Git Bash中依次执行：



```bash

\# 1. 初始化Git仓库

git init



\# 2. 添加所有文件

git add .



\# 3. 提交更改

git commit -m "Initial commit: 电力负荷预测系统"



\# 4. 连接远程仓库（替换 YOUR\_USERNAME 和 YOUR\_REPO）

git remote add origin https://github.com/YOUR\_USERNAME/YOUR\_REPO.git



\# 例如：

\# git remote add origin https://github.com/zhangsan/power-load-forecast.git



\# 5. 推送到GitHub

git push -u origin main

如果遇到权限问题，你可能需要：



bash

\# 设置用户名和邮箱

git config --global user.name "你的名字"

git config --global user.email "你的邮箱"



\# 如果提示需要验证

\# 访问：https://github.com/settings/tokens

\# 生成一个token，然后用token代替密码

git push https://你的token@github.com/用户名/仓库名.git main

