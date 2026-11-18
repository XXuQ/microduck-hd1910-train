"""Inspect and apply simulation recipes without changing upstream source files."""
import dataclasses
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

# Servo housings in the frozen Microduck robot_walk.xml, not joint counts.
# The mouth's housing is retained even though it has no walking-policy action.
SERVO_MASS_PROFILE = {
    'key': 'hd1910_23g', 'reference_g': 18.0, 'replacement_g': 23.0,
    'body_counts': {'trunk_base': 2, 'yaw2roll': 1, 'upper_leg_left': 2,
                    'leg': 1, 'neck': 2, 'yaw_roll_motion': 1, 'jaw_soft': 2,
                    'bearing_roll': 1, 'upper_leg_right': 2, 'leg_2': 1},
    'source': '用户2026-10-01确认HD1910单颗23g、XL330单颗18g；15颗含嘴巴。',
    'inertia_method': '增重部件按质量比例缩放原主惯量，重心沿用原值；这是分布近似，非重新辨识。',
}


# Separate identity preserves historical run physics and calibration snapshots.
MASS_CLOSURE_PROFILE = {
    'key': 'hd1910_820g_v1', 'target_g': 820.0,
    'title': '官方骨架 + HD1910 · 含电池820g',
    'body_additions_g': {'jaw_soft': 3.0, 'trunk_base': 4.75682},
    'source': '用户2026-10-01指定：飞特增重后头部加3g，余量加躯干，整机820g。',
}

# 834 g 实测整机配方（DuckBuddy rev-a · HD-1910 · 2S 18650）。
# 与 820g 配方不同：这里**锚定 806g 基准模型的逐 body 实测质量**（而非 737g 骨架 +
# 加克近似），再叠加三处实测修正闭合到 834g。body 集合与官方骨架完全一致
# （15 个 body 同名同结构），仅质量分布不同，因此仍可在 apply_robot 的刚体覆盖
# 框架内表达，不依赖引擎侧 microduck_hd1910 几何生成。
#
# 三处修正（源码：microduck-hd1910-train/training/hd1910/make_models.py）：
#   1) 头部缩放  256.17g -> 255.00g  （-1.17g，sh≈0.9937，分摊 neck_pitch/yaw_roll_motion/jaw_soft）
#   2) 电池      np_f970 75g -> 2S 18650 100g  （+25g，官方揉进 trunk_base，此处全部落在 trunk_base）
#   3) 铝件      trunk_base +2.05g、upper_leg_left +1.08g、upper_leg_right +1.08g  （共 +4.21g）
# 合计 -1.17 + 25 + 4.21 = +28.04g = 806 -> 834。
#
# 806g 基准 body 质量（kg，来自 upstream/robot/microduck_hd1910_base/robot_walk.xml）：
#   trunk_base 0.207691 / yaw2roll 0.0278056 / hip_l 0.00614638 /
#   upper_leg_left 0.057722 / leg 0.0263595 / ankle_left 0.0298162 /
#   neck 0.0464356 / neck_pitch 0.0056803 / yaw_roll_motion 0.0531876 /
#   jaw_soft 0.197306 / bearing_roll 0.0278056 / hip_l_2 0.00614638 /
#   upper_leg_right 0.057722 / leg_2 0.0263595 / ankle_right 0.0298167
HD1910_834G_PROFILE = {
    'key': 'hd1910_834g_v1', 'target_g': 834.0,
    'title': 'DuckBuddy rev-a · HD-1910 · 2S 18650 · 834g 实测整机',
    # 逐 body 绝对质量（kg）——锚定 806g 基准实测值，再叠加三处修正。
    # 注意：这是「绝对质量表」，不是 820g 配方那样的「加克数」。apply_robot 据此
    # 逐 body 设 mass（惯性按质量比例缩放），最终闭合到 834.0 g。
    'body_masses_kg': {
        'trunk_base': 0.23470064,                        # 0.207691 +电池25g +铝件2.05g -残差0.0404g闭合
        'yaw2roll': 0.0278056,
        'hip_l': 0.00614638,
        'upper_leg_left': 0.058802,                      # 0.057722 +铝件1.08g
        'leg': 0.0263595,
        'ankle_left': 0.0298162,
        'neck': 0.0464356,
        'neck_pitch': 0.0052903,                         # 0.0056803 -头部缩放1.17g/3
        'yaw_roll_motion': 0.0527976,                    # 0.0531876 -头部缩放1.17g/3
        'jaw_soft': 0.196916,                            # 0.197306 -头部缩放1.17g/3
        'bearing_roll': 0.0278056,
        'hip_l_2': 0.00614638,
        'upper_leg_right': 0.058802,                     # 0.057722 +铝件1.08g
        'leg_2': 0.0263595,
        'ankle_right': 0.0298167,
    },
    'source': '锚定 microduck-hd1910-train 的 806g 基准（robot_walk.xml 逐 body 实测质量）'
              ' + 三处修正（头 -1.17g / 电池 +25g / 铝件 +4.21g）= 834.0g。'
              '与 820g 配方独立并存，互不影响。',
}

# 所有「质量闭合」类配方的注册表。servo_mass_model 取这些 key 时，apply_robot 用
# 绝对 body 质量表（body_masses_kg）闭合；其余（None/xl330_18g/hd1910_23g）走原逻辑。
MASS_PROFILES = {
    MASS_CLOSURE_PROFILE['key']: MASS_CLOSURE_PROFILE,
    HD1910_834G_PROFILE['key']: HD1910_834G_PROFILE,
}


def validate_servo_mass(changes, baseline=None):
    choice = changes.get('servo_mass_model')
    if choice not in (None, 'xl330_18g', SERVO_MASS_PROFILE['key'], *MASS_PROFILES):
        raise ValueError('未知舵机重量选项。')
    if choice in (SERVO_MASS_PROFILE['key'], *MASS_PROFILES):
        if baseline not in ('official_0151',) or changes.get('mjcf_path'):
            raise ValueError('飞特增重适用于本版官方骨架；自定义MJCF请逐部件填写质量。')
        return True
    return False


def servo_mass_adjustment(mass, inertia, count, manual, extra_g=0.0):
    """Absolute manual mass takes precedence; never add the preset twice."""
    if 'mass' in manual:
        return {}
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('舵机所在刚体的原质量无效。')
    new_mass = mass + (count * (SERVO_MASS_PROFILE['replacement_g'] - SERVO_MASS_PROFILE['reference_g']) + extra_g) / 1000
    result = {'mass': new_mass}
    if 'inertia' not in manual:
        result['inertia'] = [v * new_mass / mass for v in inertia]
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_training_action_scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.01 <= value <= 2.0:
        raise ValueError('训练动作系数应为 0.01 至 2 的数字；这是动作映射，不是折扣系数 γ。')
    return float(value)


def validate_training_firmware_p(value):
    """HD1910 SRAM P is a byte; keep train/export exactly deployable."""
    if type(value) not in (int, float) or not math.isfinite(value) or int(value) != value or not 1 <= value <= 32:
        raise ValueError('HD1910舵机P应为1至32的整数；2.69是理论换算值，试训可使用最接近的3。')
    return int(value)


def plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if callable(value):
        return {"callable": getattr(value, '__module__', '') + '.' + getattr(value, '__qualname__', type(value).__name__)}
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    return {"type": type(value).__module__ + '.' + type(value).__qualname__}


def flatten(value, path=()):
    if isinstance(value, dict):
        if set(value) in ({'callable'}, {'type'}, {'slice'}):
            yield {'path': list(path), 'value': value, 'editable': False}
        elif not value:
            yield {'path': list(path), 'value': {}, 'editable': False}
        else:
            for key, item in value.items():
                yield from flatten(item, (*path, key))
    elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
        for i, item in enumerate(value):
            yield from flatten(item, (*path, i))
    else:
        blocked = {'spec_fn', 'class_type', 'func', 'json_path', 'motor_name', 'model', 'logger', 'device', 'resume', 'load_run', 'load_checkpoint', 'run_name', 'experiment_name'}
        yield {'path': list(path), 'value': value, 'editable': value is not None and not any(str(k) in blocked or str(k).startswith('_') for k in path)}


def apply_overrides(cfg, agent, patches):
    roots = {'env': cfg, 'agent': agent}
    if not isinstance(patches, list) or len(patches) > 800:
        raise ValueError('参数覆盖列表无效。')
    for patch in patches:
        keys, new = patch.get('path'), patch.get('value')
        if not isinstance(keys, list) or len(keys) < 2 or keys[0] not in roots:
            raise ValueError('参数路径无效。')
        allowed = {tuple(x['path']) for x in flatten(plain(roots[keys[0]]), (keys[0],)) if x['editable']}
        if tuple(keys) not in allowed:
            raise ValueError('不可修改的配置路径：' + str(keys))
        parent = roots[keys[0]]
        for key in keys[1:-1]:
            if isinstance(parent,dict) and key not in parent and isinstance(key,str) and key.isdigit() and int(key) in parent:key=int(key)
            parent = parent[key] if isinstance(parent, (list, tuple, dict)) else getattr(parent, key)
        key = keys[-1]
        if isinstance(parent,dict) and key not in parent and isinstance(key,str) and key.isdigit() and int(key) in parent:key=int(key)
        old = parent[key] if isinstance(parent, (list, tuple, dict)) else getattr(parent, key)
        if isinstance(old, bool):
            if type(new) is not bool: raise ValueError('需要 true / false：' + str(keys))
        elif isinstance(old, int):
            if type(new) is not int: raise ValueError('需要整数：' + str(keys))
        elif isinstance(old, float):
            if type(new) not in (float, int) or not math.isfinite(new): raise ValueError('需要有限数字。')
            new = float(new)
        elif isinstance(old, (list, tuple)):
            if not isinstance(new, list) or len(new) != len(old): raise ValueError('数组长度不匹配。')
            if any(type(v) in (int, float) and not math.isfinite(v) for v in new): raise ValueError('数组含无效数字。')
            new = tuple(new) if isinstance(old, tuple) else new
        elif isinstance(old, str):
            if not isinstance(new, str): raise ValueError('需要字符串。')
        else: raise ValueError('不支持的配置类型。')
        if isinstance(parent, dict): parent[key] = new
        elif isinstance(parent, list): parent[key] = new
        else: setattr(parent, key, new)


def apply_robot(cfg, changes, baseline=None):
    if not changes: return
    if not isinstance(changes,dict) or not set(changes)<={'bodies','joints','geoms','mjcf_path','actuator_parameters','servo_mass_model'}:
        raise ValueError('模型覆盖只支持bodies、joints、geoms、mjcf_path、actuator_parameters、servo_mass_model。')
    add_servo_mass = validate_servo_mass(changes, baseline)
    robot = cfg.scene.entities['robot']
    for index, patch in changes.get('actuator_parameters', {}).items():
        if not str(index).isdigit() or int(index)>=len(robot.articulation.actuators):
            raise ValueError('执行器组索引无效。')
        actuator=robot.articulation.actuators[int(index)]
        data=json.loads(Path(actuator.json_path).read_text())
        if not isinstance(patch,dict) or not set(patch)<=set(data):raise ValueError('BAM参数名无效。')
        for key,value in patch.items():
            if type(data[key]) not in (float,int) or type(value) not in (float,int) or not math.isfinite(value):
                raise ValueError('只支持覆盖既有BAM数值参数：'+key)
            if key in ('kt','R','armature','max_velocity','alpha','dtheta_stribeck') and value<=0:raise ValueError('BAM参数必须大于0：'+key)
            if key=='command_delay' and value<0:raise ValueError('command_delay不能为负数。')
            data[key]=value
        target=Path(tempfile.gettempdir())/'microduck-studio-bam'
        target.mkdir(exist_ok=True)
        target=target/(digest(data)+'.json')
        target.write_text(json.dumps(data,indent=2),encoding='utf-8')
        # BamActuatorCfg resolves its input path in __post_init__. Updating only
        # json_path leaves _resolved_json_path pointing at the original JSON.
        # Rebuild the dataclass so the file recorded and the file loaded agree.
        updated = dataclasses.replace(actuator, json_path=str(target))
        actuators = list(robot.articulation.actuators)
        actuators[int(index)] = updated
        robot.articulation.actuators = tuple(actuators)
    original = robot.spec_fn
    custom = changes.get('mjcf_path')
    if custom and not Path(custom).expanduser().is_file(): raise ValueError('找不到本地 MJCF 文件。')
    chosen = changes.get('servo_mass_model')
    # 只有带 body_masses_kg（绝对质量表）的配方走绝对覆盖分支；820g 仍是加克。
    absolute_profile = MASS_PROFILES.get(chosen) if chosen in MASS_PROFILES and 'body_masses_kg' in MASS_PROFILES[chosen] else None
    def build_spec():
        import mujoco
        spec = mujoco.MjSpec.from_file(str(Path(custom).expanduser())) if custom else original()
        for group, allowed in [('bodies', {'mass','pos','quat','ipos','iquat','inertia'}), ('joints', {'pos','axis','range','damping','armature','frictionloss'}), ('geoms', {'pos','quat','size','friction','contype','conaffinity','condim'})]:
            objects = {obj.name: obj for obj in getattr(spec, group)}
            for name, patch in changes.get(group, {}).items():
                if name not in objects: raise ValueError('模型中没有对象：'+name)
                if not set(patch) <= allowed: raise ValueError('模型字段不支持：'+str(set(patch)-allowed))
                if group=='bodies' and ('inertia' in patch or 'iquat' in patch):
                    objects[name].fullinertia = [float('nan'),0,0,0,0,0]
                for field, value in patch.items():
                    vals = value if isinstance(value,list) else [value]
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in vals): raise ValueError('模型参数必须是有限数值。')
                    if field in ('mass','inertia','size') and any(v <= 0 for v in vals): raise ValueError('质量/主惯量/尺寸必须大于0。')
                    setattr(objects[name], field, value)
        if absolute_profile is not None:
            # 834g 等「绝对 body 质量」配方：逐 body 设 mass（惯性按质量比例缩放），
            # 不走舵机加克。质量表锚定 806g 基准实测值，已内嵌 22.8g 舵机，故无需
            # 再叠加 SERVO_MASS_PROFILE 的 18->23g 增重。
            before = spec.compile()
            bodies = {obj.name: obj for obj in spec.bodies}
            masses = absolute_profile['body_masses_kg']
            for name, mass_kg in masses.items():
                if name not in bodies: raise ValueError('绝对质量表缺少刚体：'+name)
                bid = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, name)
                original_mass = float(before.body_mass[bid])
                body = bodies[name]
                body.mass = mass_kg
                body.ipos = before.body_ipos[bid]
                body.fullinertia = [float('nan'), 0, 0, 0, 0, 0]
                body.iquat = before.body_iquat[bid]
                body.inertia = [v * mass_kg / original_mass for v in before.body_inertia[bid]]
        elif add_servo_mass:
            # Compile after ordinary overrides, then materialize an explicit
            # inertial frame. Full-inertia XML and principal-axis XML both work.
            before = spec.compile()
            bodies = {obj.name: obj for obj in spec.bodies}
            for name, count in SERVO_MASS_PROFILE['body_counts'].items():
                if name not in bodies: raise ValueError('舵机重量映射缺少刚体：'+name)
                bid = mujoco.mj_name2id(before, mujoco.mjtObj.mjOBJ_BODY, name)
                extra_g = MASS_CLOSURE_PROFILE['body_additions_g'].get(name, 0.0) if changes.get('servo_mass_model') == MASS_CLOSURE_PROFILE['key'] else 0.0
                patch = servo_mass_adjustment(float(before.body_mass[bid]), before.body_inertia[bid],
                                              count, changes.get('bodies', {}).get(name, {}), extra_g)
                if not patch: continue
                body = bodies[name]
                body.mass = patch['mass']
                body.ipos = before.body_ipos[bid]
                if 'inertia' in patch:
                    body.fullinertia = [float('nan'), 0, 0, 0, 0, 0]
                    body.iquat = before.body_iquat[bid]
                    body.inertia = patch['inertia']
        spec.compile()  # MuJoCo validates dimensions, inertia and geometry.
        return spec
    robot.spec_fn = build_spec


def model_details(cfg):
    import mujoco
    import numpy as np
    robot = cfg.scene.entities['robot']
    spec = robot.spec_fn()
    m = spec.compile()
    bodies = []
    for i in range(1,m.nbody):
        name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)
        bodies.append({'name':name,'parent':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.body_parentid[i])),
            'mass_kg':float(m.body_mass[i]),'mass_g':float(m.body_mass[i]*1000), 'pos':m.body_pos[i].tolist(),
            'quat':m.body_quat[i].tolist(),'ipos':m.body_ipos[i].tolist(),'iquat':m.body_iquat[i].tolist(),
            'inertia':m.body_inertia[i].tolist(), 'inertia_frame':'principal axes; rotated by iquat relative to body'})
    joints = []
    init = robot.init_state.joint_pos
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i)
        if not name: continue
        home = next((float(v) for pattern,v in init.items() if re.fullmatch(pattern,name)),0.0)
        bid, did = int(m.jnt_bodyid[i]), int(m.jnt_dofadr[i])
        joints.append({'name':name,'type':int(m.jnt_type[i]),'body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bid),
            'body_mass_g':float(m.body_mass[bid]*1000),'pos':m.jnt_pos[i].tolist(),'axis':m.jnt_axis[i].tolist(),
            'range':m.jnt_range[i].tolist(),'limited':bool(m.jnt_limited[i]),'home_rad':home,'home_deg':math.degrees(home),
            'damping_xml':float(m.dof_damping[did]),'armature_xml':float(m.dof_armature[did]),'frictionloss_xml':float(m.dof_frictionloss[did])})
    geoms=[]
    for i in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)
        if name and ('collision' in name or 'foot' in name):
            geoms.append({'name':name,'body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])),
                'type':int(m.geom_type[i]),'pos':m.geom_pos[i].tolist(),'quat':m.geom_quat[i].tolist(),
                'size':m.geom_size[i].tolist(),'friction':m.geom_friction[i].tolist(),
                'contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i]),'condim':int(m.geom_condim[i])})
    entity = robot.build()
    effective = entity.spec.compile()
    actuators = []
    for index, act in enumerate(robot.articulation.actuators):
        item = plain(act)
        item['implementation'] = type(act).__name__
        p = getattr(act, '_resolved_json_path', None) or getattr(act,'json_path',None)
        if p and Path(p).is_file():
            item['parameter_file']={'name':Path(p).name,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'data':json.loads(Path(p).read_text())}
        # Read the model that the executor actually constructed, including
        # constants from the BAM actuator class that are absent from JSON.
        bam = getattr(entity.actuators[index], '_bam_model', None)
        if bam is not None:
            impl = bam.actuator
            value = lambda key: getattr(getattr(bam, key, None), 'value', None)
            vin_max = max(act.vin_range) if act.vin_range else float(impl.vin)
            kt, resistance = float(value('kt')), float(value('R'))
            pwm, current = float(impl.max_pwm), impl.max_current
            ceiling = vin_max * pwm * kt / resistance
            item['runtime_properties'] = {
                'implementation': type(impl).__name__, 'kp_fw': float(impl.kp),
                'error_gain': float(impl.error_gain), 'max_pwm': pwm,
                'max_current': current, 'kt': kt, 'R': resistance,
                'armature': float(impl.get_extra_inertia()),
                'error_gain_ratio': value('error_gain_ratio'),
                'max_velocity': value('max_velocity'),
                'duty_slope_per_rad': float(impl.kp) * float(impl.error_gain) * (value('error_gain_ratio') or 1.0),
                'electrical_damping': kt * kt / resistance,
                'compiled_force_limit_nm': vin_max * kt / resistance,
                'static_motor_ceiling_nm': min(ceiling, current * kt) if current is not None else ceiling,
                'fit_only_fields': ['q_offset', 'command_delay'],
                'note': '模型构建读回，尚未应用各环境增益/摩擦随机化；静止毛扭矩上界未扣除摩擦，不是实测堵转或连续扭矩。',
            }
        actuators.append(item)
    dynamics=[]
    for i in range(effective.njnt):
        name=mujoco.mj_id2name(effective,mujoco.mjtObj.mjOBJ_JOINT,i)
        if name:
            d=int(effective.jnt_dofadr[i])
            dynamics.append({'joint':name,'damping':float(effective.dof_damping[d]),'armature':float(effective.dof_armature[d]),'frictionloss':float(effective.dof_frictionloss[d])})
    effective_contacts=[{'name':mujoco.mj_id2name(effective,mujoco.mjtObj.mjOBJ_GEOM,i),'id':i,
        'body_name':mujoco.mj_id2name(effective,mujoco.mjtObj.mjOBJ_BODY,int(effective.geom_bodyid[i])),
        'type':int(effective.geom_type[i]),'friction':effective.geom_friction[i].tolist(),
        'condim':int(effective.geom_condim[i]),'contype':int(effective.geom_contype[i]),
        'conaffinity':int(effective.geom_conaffinity[i]),'solref':effective.geom_solref[i].tolist(),
        'solimp':effective.geom_solimp[i].tolist()} for i in range(effective.ngeom)]
    play_joints = [j for j in joints if j['name'].startswith('passive_') and j['name'].endswith('_backlash')]
    gear_backlash = {'enabled': bool(play_joints), 'joint_count': len(play_joints),
                    'ranges_rad': {j['name']: j['range'] for j in play_joints},
                    'encoder_position': 'motor + passive backlash' if play_joints else 'motor',
                    'motor_velocity': 'motor-side for friction and back-EMF',
                    'policy_joint_velocity': 'motor + passive backlash' if play_joints else 'motor'}
    return {'gear_backlash': gear_backlash, 'total_mass_kg':float(np.sum(m.body_mass)), 'bodies':bodies,'joints':joints,'collision_geoms':geoms,
        'entity_compiled':{'nu':int(effective.nu),'joints':dynamics,'geoms':effective_contacts,'note':'Entity构建后、环境随机化前；BAM动态摩擦由运行时模型计算。'},
        'actuators':actuators,'collision_overrides':plain(robot.collisions),
        'note':'质量来自选定模型。body_mass不是单颗舵机自重，父/子质量不能重复计数。XML阻尼/armature/接触设置在Entity构建时还会被BAM和collision_overrides覆盖。'}


def inspect_configs(task,cfg,agent):
    tree={'env':plain(cfg),'agent':plain(agent)}
    model=model_details(cfg)
    return {'task':task,'full':tree,'rows':list(flatten(tree)),'model':model,
            'adapter_corrections':plain(getattr(cfg, '_studio_adapter_corrections', [])),
            'sha256':digest({'full':tree,'model':model})}
