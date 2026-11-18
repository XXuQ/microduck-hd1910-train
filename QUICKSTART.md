# 快速开始（任何一台 Linux / WSL2 + NVIDIA 显卡）

目标：**克隆下来，两条命令，开始训练。**

```bash
git clone https://gh-proxy.com/https://github.com/XXuQ/microduck-hd1910-train.git
cd microduck-hd1910-train
bash setup.sh              # 约 1 分钟（引擎已缓存时）：装环境 + 生成 834g 资产 + 注册 49 个任务
bash train_auto.sh         # 先跑 1000 轮，自己判健康，正常就自动续到 10000 轮
```

`setup.sh` 跑完会打印引擎路径，并写进 `.engine_path`——之后所有脚本自动找到它，
**不需要**再手传 `--engine`。

---

## 1. 前提

| 项 | 要求 | 不满足时 |
|---|---|---|
| 系统 | Linux 或 **WSL2**（Windows 用 `wsl --install -d Ubuntu`） | macOS / Windows 原生**不支持**，MuJoCo Warp 要 CUDA |
| GPU | NVIDIA，**≥6 GB 显存**（8 GB 更稳） | 显存不够就把 `--envs` 降到 1024 |
| 磁盘 | ≥12 GB（引擎 + CUDA wheels + venv） | 建议空 20 GB |
| 软件 | `git`、`python3`、[`uv`](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| bash | **≥ 4.4**（脚本用了空数组展开） | `sudo apt install --only-upgrade bash` |

`setup.sh` 第 0 步会逐项检查，缺什么直接告诉你装什么，不会半路莫名其妙失败。

---

## 2. 卡住了看这张表

| 现象 | 原因 | 处理 |
|---|---|---|
| `Failed to connect to github.com port 443` | 国内直连不通 | 默认已走 `gh-proxy` 镜像；换镜像：`export MICRODUCK_MIRROR=https://...` |
| `uv sync` 卡住 / 超时 | PyPI 不通 | `export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` |
| `CUDA out of memory` | 并行环境太多 | `--envs 2048` 或 `1024` |
| `No module named 'torch._strobelight'` | venv 装残了（`uv sync` 被中断/磁盘满） | `rm -rf .venv && bash setup.sh`，或 `uv sync --reinstall-package torch` |
| `No space left on device` | 分区太小（`/tmp` 常见只有 8 GB） | 别把引擎装到 `/tmp`；`df -h` 确认 ≥12 GB |
| `No API key configured` | 该任务要下载教师模型做 BC 蒸馏 | 只有 VelStand 需要；本次训练用 Velocity 不涉及 |
| `^.*_servo_collision$` 匹配失败 | 资产缺舵机碰撞几何 | 重跑 `bash setup.sh`（已含补丁） |
| `unbound variable` / `bad array subscript` | bash < 4.4 | 升级 bash，或改用 `train_deploy_set.sh` |
| 训练中途消失 | WSL 被回收 / SSH 断开 | 用 `tmux`：`tmux new -s duck` → 跑 → `Ctrl-b d`，`tmux a -t duck` 回来 |
| 引擎版本漂移报错 | 官方 HEAD 变了 | `setup.sh` 会尝试切回验证过的 `0daab16`；失败会明确警告 |

---

## 3. 训练产物在哪

`.gitignore` 忽略了 `logs/`、`*.onnx`，产物**不会**进仓库：

```
<引擎目录>/logs/rsl_rl/velocity/<时间>_<run-name>/model_*.pt      # checkpoint
<引擎目录>/logs/rsl_rl/velocity/<时间>_<run-name>/<run>.onnx      # 训练结束自动导出
<仓库目录>/logs/<run-name>-s1.log                                 # 训练日志
<仓库目录>/logs/<run-name>-report.txt                             # 健康检查报告
```

远程机器训完要自己拉回：

```bash
scp -r user@server:~/microduck_rl/logs/rsl_rl/velocity/*_walk*/ ./
```

---

## 4. 常用命令

```bash
# 只看计划，不训练
bash train_auto.sh --dry-run

# 只训走路，1 万轮
bash train_auto.sh --task Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck

# 大显存卡
bash train_auto.sh --envs 4096

# 复查某份日志（不重训）
bash train_auto.sh --check-health logs/auto-s1.log

# 严格模式：reward 绝对值不达标也算失败
bash train_auto.sh --require-reward

# 换舵机刚度做对比（默认 5）
HD1910_KP=16 bash train_auto.sh --name kp16

# 按官方 6 槽部署套件逐个训
bash train_deploy_set.sh --only walk --iters 10000
```

## 5. 训练完怎么回放

```bash
cd <引擎目录>
uv run python training/hd1910/replay_onnx_bam.py --policy <onnx> --vx 0.2 --duration 8
# 出视频：加 --record walk.mp4
# 交互：  --viewer（需要图形界面；服务器上用 --record）
```

⚠ 回放脚本**没有实现低通滤波**，而 `-V6LP-` 策略是带 0.5/0.7 低通训的，
所以回放比实机更抖更激进：回放能走 → 实机大概率没问题；回放摔了 → 实机可能反而稳一些。

---

## 6. 训练和部署必须对上的三件事

| 项 | 训练侧 | 部署侧（`robotd.toml`） |
|---|---|---|
| 舵机刚度 | `HD1910_KP`（默认 **5**） | `[bus] position_kp = 5` |
| 低通 | 任务名带 `V6LP` → 0.5/0.7；不带 → 1.0 | `head_lowpass` / `legs_lowpass` 同值 |
| 动作倍率 | 训练时的 `action_scale` | `[policy] action_scale` 实测值（0.7 / 1.1 能走，0.9 / 1.0 锁死） |

对不上不会报错，只会"实机上不如仿真好看"——这是这类项目最常见的坑。
