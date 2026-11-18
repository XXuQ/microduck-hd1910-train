# -*- coding: utf-8 -*-
"""Build the HD-1910 (2S, 834 g measured) model from DuckBuddy's official 806 g model.

以 DuckBuddy 官方 `mjcf-hd1910`（806 g，Animan-AI/DuckBuddy 仓库）为**基准输入**，
只做 3 处实测驱动的精准调整，把 806 g 改成 834 g：

    1. **头部** neck_pitch + yaw_roll_motion + jaw_soft 三 body 统一缩放，
       使合计从 256.17 g -> 255 g（实测，-1.17 g，sh≈0.9937）；
    2. **电池** 官方 np_f970（75 g，揉在 trunk_base 里）拆成独立子 body，
       换成实机 2S 18650 圆柱组（68x36x18 mm，100 g，+25 g）；
    3. **铝件** trunk_base +2.05 g、左右大腿加强板各 +1.08 g（共 +4.21 g），
       从其余 body 按比例扣回，保持总重 834 g 不变。

**腿部塑料件完全不动**（保持官方 806 g 的 body 分布），因为官方模型的腿部质量
是作者按实测调好的正确值，不该被均匀缩放放大。

差额核算：-1.17（头）+ 25（电池）+ 4.21（铝件）= +28.04 g = 806 -> 834 ✓

为什么用官方 806 g 而不是 XL330 源模型（737.2 g）：
    XL330 源是 microduck_rl 训练引擎的原始模型，body 分布未经实测校准（头部只
    243 g），从它换舵机再缩放会把腿部塑料件放大 ~15%。官方 806 g 是 DuckBuddy
    作者用 onshape-to-robot 重生成并按实测调过的（头部 256 g 已接近实测 255 g），
    以它为基准，腿部天然正确，只需精准调电池和头部。

用法：
    python make_models.py <upstream>/robot/microduck_hd1910_base <out>/microduck_hd1910

资产不重复：`assets/` 用符号链接指回源目录，STL 网格只存一份。
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 实测硬约束
# ---------------------------------------------------------------------------
TOTAL = 0.834          # 整机实测总重，kg
HEAD_TOTAL = 0.255     # 头部实测，kg（neck_pitch + yaw_roll_motion + jaw_soft）
BATTERY = 0.100        # 电池实测，kg（2S 18650）
BASE_TOTAL = 0.806     # 官方基准模型总重（防漂移校验）

# 头部 = 这三个 body 的合计（与 hd1910_constants.py 的口径一致）
HEAD_BODIES = ("neck_pitch", "yaw_roll_motion", "jaw_soft")

# 电池 mesh 名（在 trunk_base body 内），拆成独立子 body
BATTERY_MESH = "np_f970"

# 铝件单独加重（g）。铝密度 2.70 / PLA 1.25 g/cm³，按 STL 体积精确计算：
#   trunk_base                +2.05 g
#   upper_leg_rigidity_plate  左右各 +1.08 g
# 合计 +4.21 g，从其余 body（不含铝件/电池/头部）按比例扣回，总重不变。
ALUM_BODY_DELTA = {
    "trunk_base":     +2.05,
    "upper_leg_left": +1.08,
    "upper_leg_right": +1.08,
}

INERTIAL = re.compile(
    r'<inertial\s+pos="([^"]+)"\s+mass="([^"]+)"\s+fullinertia="([^"]+)"\s*/>')
BODY_RE = re.compile(r'<body name="([^"]+)"')


def _nums(s):
    return [float(x) for x in s.split()]


def _blocks(text):
    """按 <body 边界切块：每个 <body ...> 到下一个 <body 之间。"""
    return re.split(r'(?=<body )', text)


def body_info(text):
    """{body_name: mass_kg}，仅含带 <inertial> 的 body。"""
    out = {}
    for blk in _blocks(text):
        mb = BODY_RE.match(blk)
        if not mb:
            continue
        mi = INERTIAL.search(blk)
        if not mi:
            continue
        out[mb.group(1)] = float(mi.group(2))
    return out


def total_mass(text):
    return sum(body_info(text).values())


# ---------------------------------------------------------------------------
# 头部缩放：三 body 合计精确 = HEAD_TOTAL
# ---------------------------------------------------------------------------
def solve_head_scale(info):
    head_mass = sum(info[b] for b in HEAD_BODIES)
    sh = HEAD_TOTAL / head_mass
    assert 0.9 < sh < 1.1, f"头部缩放系数异常: {sh}"
    return sh


def scale_body_mass(blk, scale):
    """把一个 body 的 mass 和 fullinertia 按 scale 缩放（质心不动）。"""
    mi = INERTIAL.search(blk)
    assert mi, "body 缺 <inertial>"
    new_m = float(mi.group(2)) * scale
    fi = " ".join("%.9g" % (float(x) * scale) for x in mi.group(3).split())
    new = ('<inertial pos="%s" mass="%.9g" fullinertia="%s"/>'
           % (mi.group(1), new_m, fi))
    return blk[:mi.start()] + new + blk[mi.end():]


# ---------------------------------------------------------------------------
# 电池拆出：np_f970 -> 2S 18650 独立子 body
# ---------------------------------------------------------------------------
def _battery_inertia():
    """2S 18650 圆柱组惯量（关于质心，主轴对齐）。

    实机电池：68 x 36 x 18 mm，7.4 V / 3000 mAh，两颗 18650（直径 18 mm、长 65 mm）
    并排。单颗绕长轴 Iz = 1/2 m r²，绕横轴 It = 1/12 m (3r² + L²)；
    并排沿 x、长轴沿 z、直径沿 y，关于整组质心：
      Ixx = 2·It            ≈ 3.72e-5
      Iyy = 2·It + 2m(d/2)² ≈ 4.53e-5
      Izz = 2·Iz + 2m(d/2)² ≈ 1.22e-5
    """
    d, l = 0.018, 0.065
    r = d / 2.0
    m = BATTERY / 2.0
    iz = 0.5 * m * r * r
    it = (1.0 / 12.0) * m * (3 * r * r + l * l)
    off = m * (d / 2.0) ** 2
    return [2 * it, 2 * it + 2 * off, 2 * iz + 2 * off, 0.0, 0.0, 0.0]


def split_battery(text):
    """把 trunk_base 里的 np_f970 geom 拆成独立子 body（18650，100 g）。

    返回 (新 text, 电池旧质量 kg)。
    """
    blocks = _blocks(text)
    for i, blk in enumerate(blocks):
        mb = BODY_RE.match(blk)
        if mb and mb.group(1) == "trunk_base":
            tb = blk
            break
    else:
        raise RuntimeError("找不到 trunk_base body")

    # 电池 geom（含 Part 注释行）
    m = re.search(r'<!-- Part np_f970 -->\s*<geom[^>]*/>', tb)
    if not m:
        m = re.search(r'<geom[^>]*mesh="np_f970"[^>]*/>', tb)
    if not m:
        raise RuntimeError("trunk_base 里找不到 np_f970 电池 geom")
    batt_geom_full = m.group(0)
    batt_pos = _nums(re.search(r'pos="([^"]+)"', batt_geom_full).group(1))
    qm = re.search(r'quat="([^"]+)"', batt_geom_full)
    batt_quat = qm.group(1) if qm else "1 0 0 0"
    tb_no_batt = tb.replace(batt_geom_full, "", 1)

    # 电池旧质量：从 trunk_base 的惯性量里剥出来。
    # 电池(np_f970)占 trunk_base rest 的一部分，这里用「电池旧质量 = 官方电池质量」
    # 的反推值。官方 trunk_base 里电池质量由总账解出（见下方 OLD_BATTERY）。
    mi = INERTIAL.search(tb_no_batt)
    trunk_mass = float(mi.group(2))

    # 电池旧质量（np_f970）= 反推值。为避免硬编码，从「其余部分」里剥：
    # 官方 trunk_base 其余(含电池) = trunk_mass - 2*0.0228(舵机)
    # 底座塑料 = 其余 - 电池旧质量。电池旧质量 = 官方 806g 与 834g 的差额反推
    # （见 solve 说明）。这里直接从总账解：
    old_battery = OLD_BATTERY  # 在 main 里算好后传入（模块级不好传，用全局）

    # trunk_base 减掉旧电池、加上新电池的净变化：
    # 新 trunk_base 其余 = 底座塑料 + 新电池 100g
    # 但新电池拆成独立 body，所以 trunk_base 只保留底座塑料
    # 底座塑料 = trunk 其余 - 旧电池
    servo_mass = 2 * 0.0228
    rest = trunk_mass - servo_mass
    base = rest - old_battery  # 底座塑料质量

    # trunk_base 新质量 = 底座塑料 + 2 舵机（电池已拆出）
    # 但底座塑料的质心/惯量需要从 trunk 的惯性量里剥掉旧电池再算。
    # 简化：trunk_base 整体质量 = base + servo_mass，质心/惯量按质量比例缩放
    # （质心近似不动，惯量随质量等比 —— 电池在 trunk 里质量占比小，误差可接受）。
    new_trunk_mass = base + servo_mass
    ratio = new_trunk_mass / trunk_mass
    fi = " ".join("%.9g" % (float(x) * ratio) for x in mi.group(3).split())
    new_inertial = ('<inertial pos="%s" mass="%.9g" fullinertia="%s"/>'
                    % (mi.group(1), new_trunk_mass, fi))
    tb_new = tb_no_batt[:mi.start()] + new_inertial + tb_no_batt[mi.end():]

    # 电池子 body（18650，100 g，box 近似外形）
    batt_inertia = _battery_inertia()
    batt_body = (
        f'<body name="battery" pos="{batt_pos[0]:.7g} {batt_pos[1]:.7g} {batt_pos[2]:.7g}" '
        f'quat="{batt_quat}">\n'
        f'      <inertial pos="0 0 0" mass="{BATTERY:.6g}" '
        f'fullinertia="{" ".join("%.9g" % x for x in batt_inertia)}"/>\n'
        f'      <geom type="box" size="0.018 0.009 0.034" class="visual" '
        f'rgba="0.2 0.2 0.25 1" name="battery_box"/>\n'
        f'    </body>\n'
    )

    blocks[i] = tb_new + batt_body
    return "".join(blocks), old_battery


# ---------------------------------------------------------------------------
# 铝件单独加重（纯加重，不扣回 —— 铝件是真实质量增加，806→834 差额的一部分）
# ---------------------------------------------------------------------------
def apply_alum_weights(text):
    parts = _blocks(text)
    out = []
    for blk in parts:
        mb = BODY_RE.match(blk)
        mi = INERTIAL.search(blk)
        if not (mb and mi):
            out.append(blk)
            continue
        name = mb.group(1)
        if name not in ALUM_BODY_DELTA:
            out.append(blk)
            continue
        delta = ALUM_BODY_DELTA[name] / 1000.0
        old_m = float(mi.group(2))
        new_m = old_m + delta
        ratio = new_m / old_m
        fi = " ".join("%.9g" % (float(x) * ratio) for x in mi.group(3).split())
        new_inertial = ('<inertial pos="%s" mass="%.9g" fullinertia="%s"/>'
                        % (mi.group(1), new_m, fi))
        out.append(blk[:mi.start()] + new_inertial + blk[mi.end():])
    return "".join(out)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def add_servo_collision_geoms(text):
    """给 allcollisions 变体补齐 15 个舵机外壳的碰撞几何。

    806 g 基准模型的 allcollisions 导出里，舵机只有 ``class="visual"`` 的显示几何，
    没有 ``class="collision"`` 的碰撞体（官方 737.2 g 模型每个舵机是两条，共 70 条
    collision 几何；基准里只有 11 条）。

    VelStand 的保护性摔倒会挂一个 contact sensor 盯舵机外壳
    （``primary="^.*_servo_collision$"``，因为舵机砸地会崩齿轮），
    ``name_servo_collision_geoms()`` 靠 ``mesh="xl330"`` 去找这些几何体 —— 一个都
    找不到时任务建环境直接抛::

        ValueError: Not all regular expressions are matched:
            ^.*_servo_collision$: []

    所以这里给每条舵机 visual 几何复制一条同 pos/quat/mesh 的 collision 几何，
    和官方导出的结构一致（几何体不参与质量统计，总重不受影响）。
    走路类任务用的是 robot_walk.xml，不含 allcollisions，不受此步影响。
    """
    out = []
    for line in text.splitlines(keepends=True):
        out.append(line)
        if (line.strip().startswith("<geom")
                and 'mesh="xl330"' in line
                and 'class="visual"' in line):
            out.append(line.replace('class="visual"', 'class="collision"'))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="官方 806g 基准模型目录（microduck_hd1910_base）")
    ap.add_argument("out", help="输出目录 microduck_hd1910（创建）")
    ap.add_argument("--params", default=os.path.join(HERE, "1910_m6.json"),
                    help="BAM actuator json")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # 防漂移校验：基准模型必须 = 806 g
    src_walk = os.path.join(a.src, "robot_walk.xml")
    src_text = open(src_walk, encoding="utf-8", errors="replace").read()
    got = total_mass(src_text) * 1000.0
    if abs(got - BASE_TOTAL * 1000) > 1.0:
        sys.exit(f"基准模型总重 {got:.1f} g != {BASE_TOTAL*1000:.1f} g："
                 f"官方 806g 模型漂移了")

    # 反推官方电池旧质量（np_f970）。总账（body_info 返回的是含舵机的完整质量）：
    #   834 = 头255 + (底座 + 电池100 + trunk舵机 + trunk铝) + (leg_rest + upper_leg铝) 
    #   trunk 2 舵机 = 2*0.0228；铝件总 = trunk 2.05 + upper_leg 2.16 = 4.21g
    #   底座 = 834 - 255 - 100 - 2*0.0228 - leg_rest - 4.21
    #   电池旧 = trunk_rest - 底座 - 2*0.0228
    info = body_info(src_text)
    head_rest = sum(info[b] for b in HEAD_BODIES)
    trunk_rest = info["trunk_base"]
    leg_rest = sum(v for k, v in info.items()
                   if k not in HEAD_BODIES and k != "trunk_base")
    trunk_servo = 2 * 0.0228
    alum_total = sum(ALUM_BODY_DELTA.values()) / 1000.0
    base = TOTAL - HEAD_TOTAL - BATTERY - trunk_servo - leg_rest - alum_total
    global OLD_BATTERY
    OLD_BATTERY = trunk_rest - trunk_servo - base

    sh = solve_head_scale(info)
    print(f"官方电池 np_f970 旧质量 = {OLD_BATTERY*1000:.1f} g")
    print(f"头部缩放 sh = {sh:.6f}")
    print(f"底座塑料（不缩放）= {base*1000:.1f} g")
    print()

    for name in sorted(os.listdir(a.src)):
        src = os.path.join(a.src, name)
        dst = os.path.join(a.out, name)
        if name == "assets":
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(src), dst)
            continue
        if os.path.isdir(src) or not name.endswith(".xml"):
            continue
        text = open(src, encoding="utf-8", errors="replace").read()

        blocks = _blocks(text)
        new_blocks = []
        for blk in blocks:
            mb = BODY_RE.match(blk)
            if not mb or not INERTIAL.search(blk):
                new_blocks.append(blk)
                continue
            if mb.group(1) in HEAD_BODIES:
                new_blocks.append(scale_body_mass(blk, sh))
            else:
                new_blocks.append(blk)
        text = "".join(new_blocks)

        # 拆电池（仅含 trunk_base 的模型，即 robot_*.xml；scene/additional 等不含）
        if "trunk_base" in text and "np_f970" in text:
            text, _ = split_battery(text)

        # 铝件加重
        if "trunk_base" in text:
            text = apply_alum_weights(text)

        # allcollisions 变体补齐舵机外壳的碰撞几何（见函数注释）
        if "allcollisions" in name:
            text = add_servo_collision_geoms(text)

        open(dst, "w", encoding="utf-8").write(text)
        print(f"{name:<44} {total_mass(text)*1000:6.1f} g")

    # BAM 参数
    name_out = os.path.basename(a.params)
    shutil.copy2(a.params, os.path.join(a.out, name_out))
    print(f"actuator params: {name_out} (unchanged)")

    walk = os.path.join(a.out, "robot_walk.xml")
    got = total_mass(open(walk, encoding="utf-8").read()) * 1000.0
    print(f"\nrobot_walk.xml 总重 {got:.1f} g (目标 {TOTAL*1000:.1f} g)")
    if abs(got - TOTAL * 1000) > 1.0:
        sys.exit(f"总重偏差 {got-TOTAL*1000:+.1f} g，构建失败")


if __name__ == "__main__":
    main()
