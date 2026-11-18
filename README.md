# DuckBuddy HD-1910 训练中控 + 834g 模型

自包含训练包：**kissqy 训练中控**（含 834g 质量配方）+ **完整 834g 模型资产**。
在任意 Linux + NVIDIA GPU 机器上 `git clone` 后即可开训，无需再单独拉 kissqy 官方仓库。

## 目录结构

```
.
├── console/                     # kissqy 训练中控本体（含 834g 配方改动）
│   ├── training_console.py      #   中控入口
│   ├── studio.py                #   配方/标定/模型管理
│   ├── training/                #   核心逻辑（recipe.py 含 hd1910_834g_v1）
│   ├── data/                    #   任务配置（task-configs）、baseline
│   ├── examples/                #   recipe-hd1910-834g.json 示例
│   ├── tests/                   #   单元测试（含 834g 闭合测试）
│   ├── docs/                    #   截图 + 第三方许可
│   ├── LICENSE / NOTICE.md      #   个人非商业许可 + 来源声明
│   └── ...
├── model/                       # 834g 模型资产
│   ├── hd1910/                  #   make_models.py / hd1910.py / constants / 49 任务
│   └── robot/                   #   806g 基准模型 + STL 网格资产
├── setup.sh                     # 一键搭环境：clone 引擎 + 生成 834g 资产 + 注册任务
├── train.sh / train_auto.sh     # 训练脚本
├── smoke_train.sh               # 250 轮冒烟训练
├── tools/                       # 监控 / 摩擦探测 / checkpoint 扫描
└── QUICKSTART.md
```

## 快速开始（在训练机上）

```bash
git clone https://github.com/XXuQ/microduck-hd1910-train.git
cd microduck-hd1910-train

# 1. 搭训练环境（clone microduck_rl 引擎 + 生成 834g 资产 + 注册 49 任务）
bash setup.sh

# 2. 冒烟验证
bash smoke_train.sh

# 3. 正式训练
cd ../microduck_rl
uv run list-envs | grep HD1910       # 看 49 个任务
uv run train <任务名> --env.scene.num-envs 2048
```

## 用中控调度训练（834g）

中控本体在 `console/`，含 `hd1910_834g_v1` 质量配方：

```bash
cd console
python3 training_console.py          # 启动中控（默认端口见 README）

# 在中控里选 834g 配方：robot.servo_mass_model = "hd1910_834g_v1"
# 现成示例：examples/recipe-hd1910-834g.json
```

## 834g 模型的物理口径

- **基准**：DuckBuddy 官方 806g 模型（`model/robot/microduck_hd1910_base/`）
- **三处实测修正**（`model/hd1910/make_models.py`）：
  - 头部缩放 256.17g → 255.00g（−1.17g）
  - 电池 np_f970 75g → 2S 18650 100g（+25g）
  - 铝件 +4.21g（trunk_base +2.05、upper_leg 左右各 +1.08）
- **闭合**：806 → **834.0g**（`setup.sh` 自检 `±0.05g`）
- **舵机**：飞特 HD-1910（22.8g 含舵盘实测）、`kp_fw=5`、限流 1.3A、2S 电压 6.6–8.4V

中控侧的 `hd1910_834g_v1` 配方（`console/training/recipe.py`）锚定同一套 806g 逐 body
实测质量，闭合 834.0000g，与引擎侧 `make_models.py` 生成的资产物理一致。

## 许可

- **中控原创代码**：`console/LICENSE`（个人非商业许可），保留作者署名与第三方声明。
- **834g 模型 / BAM 参数**：`model/hd1910/LICENSE-BAM-1910-LuwuDynamics`（Apache-2.0）。
- **训练引擎 microduck_rl**：Apache-2.0，由 `setup.sh` 按固定提交克隆，不在本仓库内。
- 完整来源与许可见 `console/NOTICE.md` 与 `console/docs/licenses/`。
