# 用「实测 834 g」的模型训 HD-1910 策略（手把手）

目标：训一个 **HD-1910 / 2S / 834 g** 的 walk（先走通闭环），导出成 robotd 能直接加载的 ONNX。
本目录下三个文件就是干这件事的，已经在本机用官方 XL330 资产实测过（逐刚体误差 0.000 g）：

| 文件 | 作用 |
|---|---|
| `make_models.py` | 把 microduck_rl 的 `microduck/`（XL330，737.2 g）生成 `microduck_hd1910/`（834 g） |
| `hd1910_constants.py` | 机器人配置 + 执行器（2S 电压、22.8 g 舵机、BAM 模型） |
| `hd1910_tasks.py` | 49 个任务（`...-HD1910-MicroDuck`），由 hl2915_tasks.py 机械改名生成 |

---

## 0. 你的机器实测（2026-11-15，已 ssh 读回）

`robot-collect-data.sh` 在这台鸭子上跑出的结果，已写进 `hd1910_constants.py` 和 `hd1910.py`，
原件存档在 `deploy/robots/opi-hd1910-rev-a.toml`：

| 项 | 实测值 | 训练里对应 |
|---|---|---|
| `position_kp` / `position_kp_head` | 实机测的是 **16 / 16**；**训练默认已改为 5**（见下） | `HD1910_KP = 5`（BAM 的 `kp_fw`，环境变量可覆盖） |
| `position_kd` | **8** | 拟合只在 Kd 8 下有效（和 HL-2915 一样） |
| `current_limit_ma` / `current_ma_per_count` | **1300 mA / 0.65** | `hd1910.py` 的 `DEFAULT_MAX_CURRENT = 1.3` |
| 电池 | **2S LiPo，8.004 V @ 87.8%**；**物理满电 8.4 V**（4.2×2）、空 6.6 V（3.3×2）。代码算百分比时用 4.10/节（带载），所以显示偏高约 10 个点 | `vin_range=(6.6, 8.4)`（上界就是满电 8.4）、`vin_min=6.6` |
| 控制环 | 49.99 Hz，漏拍 38 / 54574 | 50 Hz 契约没问题 |
| 温度 | 最热舵机 33 °C，CPU 61.6 °C | 训练验证时留出散热余量 |

> **2026-11-18：`HD1910_KP` 从 16 改为 5（默认）。**
>
> 依据：kissqy 的 HD-1910 行走模型（实机走通）在其部署契约里写明
> `training_physics.actuators[0].kp_fw = 5.0`，且用的是**同一份** `hd1910_m6.json`
> （`kt` 到小数点后 16 位一致）。对应刚度约 0.65 N·m/rad，是 XL330 策略（0.55）的 1.2 倍；
> 而 16 约 2.07 N·m/rad（3.8 倍），且是从 HL-2915 配方借来的——原注释就写着
> "keep it for HD-1910 until a bench fit says otherwise"。
>
> 另外 0.4–1.1 的增益随机化是**乘在 base 上**的（16×0.4=6.4），永远覆盖不到 5。
>
> ⚠ **训练和部署的 kp 必须一致**。训在 5，部署就要 `[bus] position_kp = 5`
> （robotd 侧 `gain = 5`，与 kissqy 的 `robotd-selftrained.toml` 一致）。
> 想回到 16 做对比：`HD1910_KP=16 bash train_auto.sh ...`
| 策略接口 | `obs[1,61] → actions[1,14]`（onnxruntime 实测） | manifest `model_api: 1` |

**对训练的实际影响**：Kp 和电流上限两个数现在和实机一致了（之前是猜的），
电压范围也从「文档值 6.8」改成「实测 6.6 空、8.4 满」。

### 实时状态流实测（订阅 `robot.state`，站在地上不动时）

| 关节 | 实测 | 模型 home | 差 |
|---|---|---|---|
| 左腿 5 个 | −25.8 / 2.1 / 26.9 等 | −26.2 / −0.3 / 26.0 | **全部在 ±2.4° 内** |
| 右腿 5 个 | 27.5 / −3.6 / −28.5 等 | 26.2 / 0.3 / −26.0 | 全部在 ±3.9° 内 |
| **neck** | **8.5°** | 20.0° | **−11.5°** |
| **head_pitch** | **23.7°** | 20.0° | +3.7° |
| **head_roll** | **−9.4°** | 0.0° | **−9.4°** |
| `safety.gain` | **160** | （200 × 0.8） | 站立时 Kp = round(160×16/200) = **13** |
| `policy` | **"stand"** | — | 正在跑 `hl2915_standup_v3bl_5999.onnx` |

两个结论：

1. **关节零位和符号是对的**：双腿 10 个关节全部在 home 的 ±4° 内，没有镜像错误。
2. **低头不是走路策略的问题，是「站着用的那个网络」的问题**：现在 `[policy] stand` 指向
   **HL-2915 的起身策略**，它把脖子压到 8.5°（home 是 20°）、head_pitch 抬到 23.7°，
   合起来喙朝下 **−21°**，还带 9.4° 的 head_roll 歪头。走路策略根本还没上场。

   顺带印证了 Kp 的算法：`gain = 160` 是 200 × `standing_gain_ratio 0.8`，所以站着时舵机
   Kp = 13，走路时才 16 —— 训练里 `kp_fw` 填 16 是对的（那是走路档）。

### 参考文档

> BAM 参数来源、官方训练条件、上台架辨识等辅助文档已随本仓库清理移除。
> 关键结论保留在 `hd1910_constants.py` / `hd1910.py` 的注释里（BAM 参数用公开的
> 1910 M6、`kp_fw` 16 vs 5 的取舍、2S 电压范围、限流 1.3A 等）。

### 顺手发现的三件事（都属于部署配置，不属于训练）

1. `[policy] stand = hl2915_standup_v3bl_5999.onnx` —— **HL-2915 的起身策略装在 HD-1910 上**。
   舵机力矩和整机质量都不同，摔了之后靠它起身风险很高。在 HD-1910 的 standup 训出来之前，
   建议先 `stand = "none"`（配合 `limp_fall_hand_back = false`，摔了就等人扶，别赌）。
2. `action_scale = 0.9` —— 我在 834 g 模型上扫过：0.7 和 1.1 能走，**0.9 / 1.0 会锁死不动**
   （步频 0.17 Hz，与质量无关）。实机可能因为别的原因能用，但值得做一次 A/B：0.7 vs 0.9 vs 1.1。
3. `sitstand = "none"` —— 和之前结论一致：这台车现在**不会坐**。

---

## 0.5 先看你的机器够不够

| 项 | 最低 | 舒服 |
|---|---|---|
| GPU | **NVIDIA，≥8 GB（RTX 3060/4060）** | 16–24 GB（4080/4090） |
| 并行环境 | `--num_envs 1024`（8 GB） | 4096（作者用 V100） |
| 磁盘 | 30 GB（MuJoCo Warp + 缓存） | 80 GB |
| 系统 | Linux（Windows 只能跑 MuJoCo 回放，训不了） | Ubuntu 22.04 + CUDA 12 |

没有 8 GB N 卡就别硬上：按小时租（AutoDL / RunPod），作者自己也是 `ssh v100` 远程训。

---

## 1. 装 microduck_rl

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
uv sync            # 官方用 uv；它会把 mujoco-warp / rsl_rl / onnx 都装好
uv run python -c "import mjlab, rsl_rl; print('ok')"
```

---

## 2. 生成 834 g 的 HD-1910 资产

把这个仓库的 `training/hd1910/` 整个拷进 microduck_rl 的工作目录，然后：

```bash
cd microduck_rl
python training/hd1910/make_models.py \
    src/mjlab_microduck/robot/microduck_hd1910_base \
    src/mjlab_microduck/robot/microduck_hd1910
```

输出应该是这样（已实测）：

```
官方电池 np_f970 旧质量 = 75.0 g
头部缩放 sh = 0.995418
底座塑料（不缩放）= 87.1 g

robot_walk.xml                       834.0 g
...
robot_walk.xml total 834.0 g (target 834.0 g)
```

**它在做什么**（以 DuckBuddy 官方 806g 模型为基准，只做 3 处实测驱动调整）：

1. **头部**（neck_pitch + yaw_roll_motion + jaw_soft 三 body）统一缩放，使合计从
   官方 256.17 g → 实测 **255 g**（sh = 0.9954，-1.17 g）；
2. **电池** 官方 np_f970（75 g，揉在 trunk_base 里）拆成独立子 body，换成实机
   2S 18650 圆柱组（68×36×18 mm，**100 g**，+25 g），惯量用圆柱解析解；
3. **铝件** trunk_base +2.05 g、左右大腿加强板各 +1.08 g（共 +4.21 g）。

**腿部塑料件完全不动**（保持官方 806g 的 body 分布）——因为官方模型的腿部质量
是作者按实测调好的正确值，不该被均匀缩放放大。

差额核算：-1.17（头）+ 25（电池）+ 4.21（铝件）= +28.04 g = 806 → 834 ✓

> 校验：生成结果里 `battery` 子 body 应 = **100.00 g**、头部三 body 合计 = **255.00 g**、
> 总重 = **834.0 g**、腿部 body 与官方 806g 完全一致。对不上就是基准模型漂了
> （脚本会直接报错退出）。

---

## 3. 装 constants 和 tasks

```bash
cp training/hd1910/hd1910_constants.py src/mjlab_microduck/robot/
cp training/hd1910/hd1910_tasks.py     src/mjlab_microduck/tasks/
cp training/hd1910/1910_m6.json        src/mjlab_microduck/robot/microduck_hd1910/  # BAM 参数（make_models 也会带）
echo 'from mjlab_microduck.tasks import hd1910  # noqa: F401' >> src/mjlab_microduck/tasks/__init__.py

uv run python -c "
from mjlab.tasks.registry import list_tasks
print([t for t in list_tasks() if 'HD1910' in t][:5])"
```

能看到 `Mjlab-Velocity-Flat-Backlash-LowFriction-V6LP-HD1910-MicroDuck` 就对了。

---

## 4. 训第一版 walk

先跑**最贴 HL-2915 上走得最好的那条配方**（`v6lp_slip` = 低摩擦 + 模拟低通）：

```bash
uv run train Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck \
    --num_envs 2048 --max_iterations 4000
```

> 命令形状取决于你的 microduck_rl 版本（`uv run train <task>` 或 `uv run python scripts/train.py <task>`），
> 以仓库 README 为准；参数名是 mjlab 的 `--num_envs` / `--max_iterations`。

时间参考（作者 V100、4096 env、3999 轮）：**一版 walk 几个小时**。8 GB 卡按 1024–2048 跑，翻 2–4 倍。

训练时**别停在中途**：3999 轮之前步态还没收敛，`checkpoint_sweep` 会挑到看着能走、上机就摔的那一版。

---

## 5. 导出 ONNX（CPU 就行）

```bash
# 1) 先让 export.py 给同一任务导一个模板（唯一需要 GPU 的一步，normalizer 烘进图里）
uv run python scripts/export.py --task Mjlab-...-HD1910-MicroDuck --checkpoint <run>/model_4000.pt

# 2) 之后所有 checkpoint 都用 ckpt_onnx.py 换权重（纯 CPU，秒级）
CUDA_VISIBLE_DEVICES= .venv/bin/python training/hl2915/ckpt_onnx.py \
    --template <run>/<run>.onnx --out-dir sweep/ --prefix hd1910_walk_ <run>/model_3999.pt
```

导出的模型必须是 `obs[1,61] → actions[1,14]`，且**观测归一化已烘进模型**——官方 0.21 明确要求
（`docs/recurrent-policies.md`：「Observation normalization must be baked into the exported model」）。

---

## 6. 写 manifest.json（官方 0.20+ 的硬约束）

官方现在**先读 manifest 再下载**，`obs_len` / `action_len` / `model_api` / `robot.model` 任一对不上就直接拒：

```json
{
  "schema_version": 2,
  "model_api": 1,
  "obs_len": 61,
  "action_len": 14,
  "robot": { "model": "microduck", "hw_rev": 1, "servos": "hd1910", "control_hz": 50 },
  "name": "hd1910-walk",
  "kind": "perpetual",
  "slot": "walk",
  "action_scale": 0.7,
  "command": { "encoding": "constant", "idle": [0, 0, 0],
               "twist": "vx, vy, vyaw (m/s, m/s, rad/s)", "head": "4 slots, offsets from home",
               "body": "z, roll, pitch" },
  "training": { "task_id": "Mjlab-Velocity-Flat-Backlash-Slippery-V6LP-HD1910-MicroDuck",
                "repo": "pollen-robotics/microduck_rl", "checkpoint": 3999 }
}
```

`action_scale` 写多少要**实测**：834 g 模型上扫过，**0.7 和 1.1 能走，0.9/1.0 会锁死不动**
（策略固有特性，与质量无关）。你机器上现在跑的是 **0.9**，训完新策略别直接照抄，按实测填。

---

## 7. 部署到鸭子

```toml
# deploy/robotd.toml 的 [policy]
walk = "/home/orangepi/policies/hd1910_walk_v6lp_slippery_3999.onnx"

# 滤波必须和训练一致：-V6LP- 训练时模拟了低通 → 部署也用 0.5 / 0.7
head_lowpass = 0.5
legs_lowpass = 0.7
```

训的是不带 LP 的版本（任务名里没有 `V6LP`）就把两个滤波都设 **1.0**——混用会让策略抖或迟钝，
官方 `docs/recurrent-policies.md` 专门写了一条。

```bash
systemctl --user restart robotd-dev
~/robot/robotctl health   # 或者用 robot-look.py --status
```

---

## 8. 上机前先在仿真里回放

**不要用 `training/sim_walk.py`**：它用简单 position 执行器，没有 BAM 力矩/电流模型，
同一个 onnx 在它下面几乎原地不动（5 s 前进 ~2 mm）。用本仓库的 `replay_onnx_bam.py`
（BAM + HD-1910 拟合参数，`setup.sh` 会一起装到引擎目录）：

```bash
cd ~/microduck_rl

# 无头跑 8 秒，直接出指标（前进距离 / 稳态速度 / roll-pitch / 是否摔倒）
uv run python training/hd1910/replay_onnx_bam.py \
    --policy <新 onnx> --scene src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml \
    --vx 0.2 --duration 8

# 出 960x720 视频（不经过 WSLg，播放最顺）
uv run python training/hd1910/replay_onnx_bam.py --policy <新 onnx> \
    --vx 0.2 --duration 8 --record walk.mp4

# 交互式（WASD 控、空格停、P 推一把）
uv run python training/hd1910/replay_onnx_bam.py --policy <新 onnx> --viewer
```

看四件事：能不能走满 8 秒不摔、`vx 稳态`是否接近指令的 50% 以上、
roll/pitch RMS 是否 < 5°、侧漂 Δy 是否可接受（策略有固有偏航，见 11.6）。

模型几何/舵机角度存疑时，先跑 `joint_viewer.py --list` / `--sweep`（见 11 节）。

---

## 9. 已知的坑（按重要性）

1. **执行器模型用的是公开的 1910 M6**（LuwuDynamics，Apache-2.0，SHA-256 `ef2d51ad…`），
   不是 HL-2915 的借用品。`"actuator": "sts3215"` 决定用 BAM 自带的飞特控制律，别改这个字段。
   上游训练用 `kp_fw = 5`，我们**已改成 5**（2026-11-18，依据 kissqy 实机配置，见第 1 节表格下方说明）。
   部署必须同步 `[bus] position_kp = 5`；想对比 16 就 `HD1910_KP=16` 重训一版。
2. **2S 电压写死了**。`hd1910_constants.py` 里 `vin_range=(6.6, 8.4)`、`vin_min=6.6`；照抄 HL-2915 的
   10.0–12.8 会让策略以为电池永远满电。
3. **低头问题**：现有 HD-1910 策略的喙比水平低 24°，因为基础配方没有 `-HeadTrack-`。想修就训
   `...-HeadTrack-...`（`_head_track()` 已写好，std 0.3 / weight 3.0）。
4. **别用 `--num_envs` 太小**。作者注释里 0.4–1.1 的增益随机化需要足够多的环境才吃到，
   低于 1024 容易训出「只在一种增益下能走」的策略。
5. **hd1910_tasks.py 是机械生成的**。以后再改 `hl2915_tasks.py`，用同样方式重新生成，
   别手改两份（仓库自己的规矩就是「一份真相」）。

---

## 10. 最短路径建议

1. 先只训 **walk**（第 4 节），跑通「训练 → 导出 → manifest → 部署 → 回放」整条链；
2. 走路稳定后再补 **standup**（HD-1910 现在完全没有起身策略，摔了自己起不来）；
3. 最后才碰 sitstand / roulade —— 它们要额外的姿态目标与接触力奖励，参数多，别一步到位。

---

## 11. WSL2 本地验证（2026-11-17 实测通过）

在 WSL2 Ubuntu + RTX 2060 6GB 上完整验证过「搭环境 → 生成 834g → 加载任务 → 回放」。

### 11.1 两个必须知道的坑

1. **`STS3215Actuator` 在 mjlab 路径下起不来**。`1910_m6.json` 的
   `"actuator": "sts3215"` 选中的 BAM 类，其 `q_target_smooth` 只在台架
   `load_log()` 里创建；mjlab 训练/回放从不调用它 → 第一次 `compute_control()`
   抛 `AttributeError: 'STS3215Actuator' object has no attribute 'q_target_smooth'`。
   **修法**：`hd1910_constants.py` 末尾已加惰性初始化补丁（保持拟合律不变）。
   ⚠️ 不要改用 `"actuator": "xl330"` 绕过——两者 `error_gain` 差 58 倍
   （0.166 vs 0.00288），换了等于换一套控制律。

2. **`hd1910_tasks.py` 拷进 tasks/ 后必须改名 `hd1910.py`**。
   `cp x/hd1910_tasks.py src/.../tasks/` 会保留原名，而 ``tasks/__init__.py`` 里是
   `from mjlab_microduck.tasks import hd1910` → 循环导入报
   `cannot import name 'hd1910' from partially initialized module`。
   （`setup.sh` 里这条也得跟着改名。）

### 11.2 回放必须用 BAM，不能用 sim_walk

DuckBuddy 的 `training/sim_walk.py` 用**简单 position 执行器**，策略在它下面几乎原地不动
（5 s 前进 ~2 mm，官方 806g 模型上也一样）。用本仓库的
`replay_onnx_bam.py`（BAM HD-1910 执行器）在 microduck_rl 根目录跑：

```bash
# 无头跑 8 秒出指标
uv run python training/hd1910/replay_onnx_bam.py \
    --policy <hd1910_walk_kp16_6500.onnx> \
    --scene  src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml \
    --vx 0.15 --duration 8

# 打开 MuJoCo 可视化窗口，键盘实时控制（默认原地站立，按 W 开始走）
uv run python training/hd1910/replay_onnx_bam.py \
    --policy <hd1910_walk_kp16_6500.onnx> \
    --scene  src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml --viewer
```

**可视化键盘控制**（先点一下 3D 窗口给它焦点）：

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `W` / `↑` | 前进 +0.05 m/s | `S` / `↓` | 后退 / 减速 |
| `A` / `←` | 左移 +0.05 m/s | `D` / `→` | 右移 |
| `Q` | 左转 +0.2 rad/s | `E` | 右转 |
| `空格` | **立即停止** | `R` | 重置到 STAND 姿态 |
| `H` | 打印帮助 | | |

可视化模式**默认 vx=0（原地站立）**，不会像旧版那样一开窗就走没影；
要给初始速度用 `--vx 0.15`。

`--viewer` 需要显示环境（WSL2 有 WSLg 即可）；`--render-hz`（默认 30）控制渲染频率，
`--speed`（默认 1.0）给仿真加速/慢放，`--profile` 打印策略/物理/渲染分段耗时。
**`--record out.mp4`** 把回放离屏渲染成视频文件（`--record-fps` 默认 30），
播放完全不走 WSLg 传输，WSL 下看回放最顺。
对比 806 g 基准把 `--scene` 换成
`src/mjlab_microduck/robot/microduck_hd1910_base/scene_walk.xml`。

```bash
# 录一段对比视频（不需要窗口，几秒出片）
uv run python training/hd1910/replay_onnx_bam.py \
    --policy <hd1910_walk_kp16_6500.onnx> \
    --scene  src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml \
    --vx 0.2 --duration 6 --record walk834.mp4
```

### 11.4b 可视化交互：控制 + 推它（外力扰动）

```bash
cd ~/microduck_rl && uv run python training/hd1910/replay_onnx_bam.py \
  --policy <hd1910_walk_kp16_6500.onnx> \
  --scene src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml --viewer
```

窗口左上角有 HUD，实时显示当前指令；**窗口获得焦点后按键**（也可直接在终端按）：

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `W`/`↑` | 前进 +0.05 m/s | `S`/`↓` | 后退/减速 |
| `A`/`←` | 左移 +0.05 m/s | `D`/`→` | 右移 |
| `Q` | 左转 +0.2 rad/s | `E` | 右转 |
| `空格` | 立即停止 | `R` | 重置姿态 |
| **`P`** | **随机推它一把 0.8–1.5 N、0.15 s** | `H` | 打印帮助 |

**用鼠标推**：在 MuJoCo 窗口里按住 **Ctrl + 左键拖动**，即可在鸭子身上施加力（点哪儿推哪儿，
拖得越远力越大）——这是 MuJoCo 自带的扰动功能，不需要改代码。

**无人值守测抗扰**：`--auto-push 3` 会在第 3 秒自动推一把，方便批量跑"推了还站不站得住"。

实测（834 g、vx 0.15 + vyaw 0.15、第 3 秒推）：

| 扰动 | 结果 |
|---|---|
| 1.46 N / 227° | 侧移 0.37 m，**未摔倒** |
| 0.82 N / 215° | 侧移 0.29 m，**未摔倒**（见 `render_out/push834.mp4`）|

### 11.5 用官方教程的方法测试（`infer_policy.py --new-cmd-obs`）会摔

官方教程（microduck 完整训练教程）给的测试方式是
`uv run scripts/infer_policy.py --walking <onnx> --new-cmd-obs`。
**直接用它跑 HD-1910 策略会摔**（834 和 806 都是 1 秒内 trunk_z 从 −67 → −110 mm 趴地）：

```
BAM m6 actuators on 14 joints: kt=0.3660 R=2.8114 ...      ← xl330 参数
```

`infer_policy.py` 的 `load_bam_model()` 写死了内置 **xl330** M6 模型，
而 HD-1910 的拟合是 **kt=0.6238 / R=4.9102**（`1910_m6.json`）。
力矩估算差约 1.7 倍，鸭子撑不住自己。

**用 `training/hd1910/infer_policy_hd1910.py` 替代**（同入口、同参数，只换 BAM 来源）：

```bash
uv run python training/hd1910/infer_policy_hd1910.py \
    --walking <hd1910_walk_kp16_6500.onnx> --new-cmd-obs \
    --scene src/mjlab_microduck/robot/microduck_hd1910/scene_walk.xml \
    --lin-vel-x 0.2 --kp-fw 16 --vin 7.4 --current-limit 1.3
```

实测改成 1910 参数后 834 / 806 **都不摔**，能稳定行走。终端控制键与官方一致
（方向键上/下前进后退、A/E 转、空格清零）。

### 11.6 「走不直」是策略固有偏航，不是模型问题

实测（vx 指令 0.2、yaw 指令 0，1 秒平均）：

| 模型 | 偏航实测 | 说明 |
|---|---|---|
| 我们 834 g | −0.06 ~ −0.48 rad/s | 一直朝一个方向漂 |
| 官方 806 g | −0.03 ~ −0.31 rad/s | **同样漂、量级一致** |

- 左右件质量完全相等（ankle 29.82/29.82、upper_leg 58.80/58.80、hip 6.15/6.15），排除模型不对称；
- DuckBuddy 自己的 `training/policies/README.md` 就写了：同一次训练各 checkpoint
  直走偏航 **−30 ~ +42 °/s** 不等，他们靠"按偏航扫描挑 checkpoint"选模型 →
  这是**策略本身的 yaw bias**；
- 前进也只跑到 0.09–0.13（指令 0.2），同一个原因：跟踪不足。

**临时缓解**：给一点反向偏航指令就能走直（实测 834 g、vx=0.2）：

| `--ang-vel-z` | 实测 yaw |
|---|---|
| 0（不加） | −0.06 ~ −0.48 rad/s（右漂）|
| **+0.20** | +0.03 ~ +0.14 rad/s（基本走直）|
| +0.30 | +0.04 ~ +0.20 rad/s |

**根治**：重训并加 yaw-rate 惩罚/对称损失（仓库里有 `tasks/symmetry.py`），
或按偏航扫描挑 checkpoint；实机上可开方向保持。

实测（官方 `hd1910_walk_kp16_6500.onnx`，action_scale=1.0，8 s）：

| 模型 | vx=0 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| **我们 834 g** Δx | +0.01 m | +0.00 m | +0.43 m | +0.51 m | +1.15 m |
| 官方 806 g Δx | — | — | +0.32 m | — | +1.28 m |
| 摔倒 | 无 | 无 | 无 | 无 | 无 |

- 0 指令站立极稳（roll/pitch RMS 0.1°），0.10 m/s 策略倾向站定（死区），0.20 起明显起步；
- 834 g 与 806 g 行为一致、都不摔，姿态 RMS 差异 < 0.5°（电池 +25 g 没有破坏策略）。

### 11.3 训练环境本身（GPU）也验证过
```bash
uv run python -c "
import torch; from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, list_tasks
t=sorted(x for x in list_tasks() if 'HD1910' in x and 'Velocity' in x)[0]
c=load_env_cfg(t, play=True); c.scene.num_envs=4
e=ManagerBasedRlEnv(cfg=c, device='cuda:0'); o,_=e.reset()
[e.step(torch.zeros(e.action_space.shape, device='cuda:0')) for _ in range(200)]
print('ok', o['actor'].shape)"   # -> ok torch.Size([4, 61])
```

`obs[·,61] → action[·,14]`，与 robotd 契约一致；`robot_walk.xml` 总质量 834.0 g。

### 11.4 国内网络：`uv run` 卡住几十秒（实测 63 s → 0.02 s）

**症状**：敲完命令只打出 `colorama==0.4.6` 之类的同步行，几十秒没有任何 BAM 日志，
看起来像"没反应"。**不是回放脚本卡住，是 `uv run` 在联网校验。**

**根因**：`bam` 是 **git 依赖**（`git+https://github.com/Rhoban/bam.git@mjlab_frictionloss`），
`uv run` 每次都会查这个分支有没有更新；本机连 github.com 单次握手 ~19 s，重试几次就是 60 s+。
跟 PyPI 无关（PyPI 换镜像解决不了它）。

**三处配置（已验证）**：

```bash
# 1) uv 默认索引换清华（首次 uv sync 用；torch 仍走 pyproject 里的 cu129 索引）
mkdir -p ~/.config/uv && cat > ~/.config/uv/uv.toml <<'TOML'
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
[[index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple"
http-timeout = 120
network-concurrency = 16
TOML

# 2) 只给 Rhoban 组织挂 GitHub 加速（不影响其它仓库的 git push）
git config --global url."https://gh-proxy.com/https://github.com/Rhoban/".insteadOf \
                   "https://github.com/Rhoban/"

# 3) 克隆 microduck_rl 本身也走镜像（4 s）
git clone --depth 1 https://gh-proxy.com/https://github.com/pollen-robotics/microduck_rl.git
```

**兜底**：环境已经装好时，`uv run --no-sync python ...` 完全跳过联网校验（0.02 s），
急着看回放时最省事。不要全局 `export UV_NO_SYNC=1` —— 会让以后的 `uv sync` 也变成空操作。

### 11.7 可视化"卡"的真正原因：Mesa 退回软件渲染（llvmpipe）

**结论先说**：不是 WSLg 传输，是 **GPU 没被用上**，MuJoCo 一直在 CPU 软渲染。

```text
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
GL: Mesa / llvmpipe (LLVM 21.1.8, 256 bits)       ← 软件渲染
render 单帧: 321 ms  (640x480)
```

两个原因叠加：

1. `/dev/dri/renderD128` 属于 `root:render`，当前用户不在 `render` 组 → 打不开 GPU 节点；
2. 即使有权限，Mesa 也默认选了 `kms_swrast`，要显式指定 d3d12 后端。

**修复（已在本机做完，重装系统后要重做）**：

```bash
# 1) 让用户能访问 GPU 节点（root 身份执行，不需要密码）
wsl.exe -d Ubuntu -u root -- usermod -aG render,video $USER
#    然后在 Windows 侧 wsl --terminate Ubuntu 重启，组权限才生效

# 2) 强制 Mesa 走 WSLg 的 d3d12 后端（已写入 ~/.bashrc）
export MESA_LOADER_DRIVER_OVERRIDE=d3d12
export GALLIUM_DRIVER=d3d12
```

效果对比（640×480 离屏单帧）：

| | GL 后端 | 单帧耗时 |
|---|---|---|
| 修复前 | Mesa / llvmpipe（软渲染）| **321 ms** |
| 修复后 | Microsoft / D3D12 (RTX 2060) | **3.0 ms** |

**快 107 倍**。8 秒、960×720、50 fps 的录制从 **5 分钟降到 1.7 秒渲染 + 编码**（整条命令约 5 秒）。

**不需要装 Ubuntu 虚拟机**：VM 里 GPU 直通基本不可用，MuJoCo Warp 要 CUDA，只会更慢。

> 实测窗口模式仍有约 79% 实时率（sync 5.3 ms/帧的 Python 循环开销），
> 视觉上是"慢放 0.8 倍"而不是卡顿；纯看回放建议用 `--record` 出 mp4。
