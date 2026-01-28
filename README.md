# PolyFlow

# 该代码是 PolyFlow: Safety Guaranteed and Sample Efficient Flow Matching via Constraint Embedding and Projection-free Update ICML2026 投稿论文的代码

## 框架介绍

TODO: 根据pdf文章简要介绍方法，并且给出figs/文件夹下面的两张图（并排放置）

## 依赖安装

该项目基于 python==3.10, 建议新建conda环境，并基于 requirements.txt 安装依赖

## 运行代码

我们将maze2d任务和locomotion任务分别放到 maze 和 locomotion 两个分支下。

如果想运行locomotion任务，执行下面命令
```
git switch locomotion
```

要训练PolyFlow模型，执行下面命令：
```
python train_polyflow.py --config-name=train_polyflow_xxxxxx_fixcons.yaml
```
其中 xxxx 填入不同的任务名称：hoppercpx: Hopper-Simple; hoppercpx2: Hopper-Complex; walkercpx: Walker2d-Simple; walkercpx2: Walker2d-Complex

训练结果会保存到 outputs/ 文件夹下

如果想进行rollout，执行下面命令：
```
python sample_polyflow.py --config-name=time_polyflow_xxxxxx_fixcons.yaml eval.load_model_path=yyyyy
```
其中 yyyy 处填入对应的模型保存路径



# Acknowledgements

其中有关diffusion和safediffuser的代码参考了 https://github.com/jannerm/diffuser 和 https://github.com/Weixy21/SafeDiffuser 两个项目。
