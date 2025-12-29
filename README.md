# PolyFlow

## 框架介绍

* env: 环境交互类
* dataset: 数据集类
* backbone: 模型架构
* algorithm: 方法类（diffusion, flow）
* trainer: 训练类
* policy: 策略类

## 如何运行？

### diffusion
* 训练 diffusion 
```bash
python train_diffusion.py --config-name=train_diffusion_hopper.yaml
```

* 训练 diffusion classifier
```bash
python train_diffusion_value.py --config-name=train_diffusion_value_hopper.yaml
```

* 采样
```bash
python sample_diffusion.py --config-name=sample_diffusion_hopper.yaml
```

* guided 采样
```bash
python sample_diffusion_guide.py --config-name=sample_diffusion_guide_hopper.yaml
```

### flow
* 训练 flow
```bash
python train_flow.py --config-name=train_flow_hopper.yaml
```

# Acknowledgements
The diffusion model implementation and organization are based on Michael Janner's diffuser repo: https://github.com/jannerm/diffuser
