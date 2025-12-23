

# Poly-Flow

## 依赖安装
项目依赖 `python==3.10`
```
pip install -r requirements.txt
```

## 如何运行？

* 如果想训练模型, 运行下面指令
```
python train.py --config-name="train_oneray_maze2d.yaml" device="cuda:0"
```
所有训练参数放在 config 中的 yaml 文件内

所有训练过程的日志，模型文件和评估指标放在 outputs 文件夹内

* 如果只想跑采样过程，运行下面指令
```
python sample.py --config-name="sample_safeflow_maze2d.yaml" device="cuda:0"
```

同样地，采样的评估指标放在 outputs 文件夹内。