import blenderproc as bproc          # ← must be first!
import argparse
import csv
import imageio
import numpy as np
import mathutils
from pathlib import Path
import shutil
from blenderproc.python.writer.WriterUtility import _WriterUtility

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj_path",            required=True)
    ap.add_argument("--obj_texture_path",    required=True)
    ap.add_argument("--obj_texture_mtl_path",    required=True)
    ap.add_argument("--pose_csv",            required=True,
                    help="CSV with headers Rollout_step,x,y,z,qw,qx,qy,qz")
    ap.add_argument("--obj_scale",           nargs=3, type=float, default=[1,1,1])
    ap.add_argument("--box_scale",           nargs=3, type=float, default=[0.3*1.5,0.3*1.5,0.05])
    ap.add_argument("--color_out_dir",       required=True,
                    help="Directory for color images")
    ap.add_argument("--depth_out_dir",       required=True,
                    help="Directory for depth images")
    ap.add_argument("--mask_out_dir",        required=True,
                    help="Directory for mask images (only step 1)")
    ap.add_argument("--bg_path",             required=True)
    ap.add_argument("--box_tex_path",        default = None)
    ap.add_argument("--K_file",              required=True)
    args = ap.parse_args()

    # prepare output dirs
    color_dir = Path(args.color_out_dir); color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = Path(args.depth_out_dir); depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir  = Path(args.mask_out_dir);  mask_dir.mkdir(parents=True, exist_ok=True)
    for d in (color_dir, depth_dir, mask_dir):
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    # read all poses from CSV
    poses = []
    with open(args.pose_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                "step": int(row["Rollout_step"]),
                "loc":  [float(row[k]) for k in ("x","y","z")],
                "quat": [float(row[k]) for k in ("qw","qx","qy","qz")],
                "cyl_loc":  [float(row[k]) for k in ("fx","fy","fz")],
            })

    # --------------------------------------------------------------------------
    bproc.init()
    # container
    boxes = create_box(args.box_scale, tex_path=args.box_tex_path)
    for b in boxes:
        b.set_cp("category_id", 0)

    # load single object once
    obj = load_obj_with_materials(
        args.obj_path,
        texture_path=args.obj_texture_path,
        mtl_path=args.obj_texture_mtl_path,
        scale=args.obj_scale
    )
    obj.set_cp("category_id", 1)
    
    cyl = make_cylinder(0.01,0.03)
    cyl.set_cp("category_id", 2)

    # camera
    cam_euler = (np.radians(40), 0, 0)
    cam_loc   = [0, -0.9, 1]
    setup_cam(cam_euler, cam_loc, resolution=[512,512])

    # lighting & background
    # setup_indoor_lighting(bg_path=args.bg_path)
    setup_area_lighting([0,0,3], bg_path=args.bg_path)

    # intrinsics
    W, H = 512, 512
    K = np.loadtxt(args.K_file, dtype=np.float32)
    bproc.camera.set_intrinsics_from_K_matrix(K, W, H)

    # enable depth output (so data["depth"] appears)
    bproc.renderer.enable_depth_output(
        activate_antialiasing=True,
        output_key="depth",
        convert_to_distance=True,
        antialiasing_distance_max=10.0
    )
    
    bproc.renderer.enable_segmentation_output(map_by=["instance","name"])
    
    # loop over every pose
    # div = 15
    # for i in range(1,int(np.ceil(len(poses)/div))):
    for i in range(1,len(poses)):
    # for i in range(3,10):
        # set object pose
        idx = i
        # idx = i*div - div + 1
        print(f"rendering rollout {idx}")
        p = poses[idx]
        obj.set_location(p["loc"])
        # convert quaternion to Euler for BlenderProc
        q = mathutils.Quaternion((p["quat"][0], p["quat"][1], p["quat"][2], p["quat"][3]))
        eul = np.array(q.to_euler("XYZ"))
        obj.set_rotation_euler(eul)
        
        raw_x, raw_y, raw_z = p["cyl_loc"]
        # apply your trial‐and‐error offsets
        x_offset = -0.015/2
        y_offset = 0.015/2
        corrected_loc = [raw_x + x_offset, raw_y + y_offset, raw_z]
        cyl.set_location(corrected_loc)

        # render
        data = bproc.renderer.render()
        rgb   = data["colors"][0]
        depth = data["depth"][0]
        inst_map = data["instance_segmaps"][0]
        attr = data["instance_attribute_maps"][0]

        # write out color
        color_path = color_dir / f"color_{idx}.png"
        imageio.imwrite(color_path, rgb)
        print(f"→ {color_path}")

        # convert to mm and write depth
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_path = depth_dir / f"depth_{idx}.png"
        imageio.imwrite(depth_path, depth_mm)
        print(f"→ {depth_path}")

        # mask only on first step
        if idx == 1:
            my_idx = next(m["idx"] for m in attr if m["name"] == obj.get_name())
            mask   = (inst_map == my_idx).astype(np.uint8) * 255
            mask_path = mask_dir / "mask_001.png"
            imageio.imwrite(mask_path, mask)
            print(f"→ {mask_path}")
        

def load_obj_with_materials(obj_path, mtl_path=None, texture_path=None, scale=None):
    obj_path = Path(obj_path)
    # if user gave a .mtl, copy it so OBJ’s mtllib line will find it
    if mtl_path:
        dest = obj_path.with_suffix('.mtl')
        shutil.copy(mtl_path, dest)

    # load; BlenderProc will import materials from that .mtl if present
    objs = bproc.loader.load_obj(str(obj_path))
    obj  = objs[0]
    if scale is not None:
        obj.set_scale(scale)

    # if no .mtl, but a single texture was specified, override materials
    if not mtl_path and texture_path:
        mat = bproc.material.create_material_from_texture(
            texture_path, material_name="tex_mat")
        obj.replace_materials(mat)

    return obj

def make_cylinder(radius, height, name="cylinder"):
    """
    Create a cylinder of given radius & height, resting on Z=0.
    """
    # center it so bottom is at z=0
    z_center = height / 2.0

    cyl = bproc.object.create_primitive(
        "CYLINDER",
        scale=[radius, radius, height],    # <— three values only
        location=[0.0, 0.0, z_center],      # <— three values only
        rotation=[0.0, 0.0, 0.0]            # (optional) three Euler angles
    )
    cyl.set_name(name)

    # give it a simple grey material
    mat = bproc.material.create(name + "_mat")
    mat.set_principled_shader_value("Base Color", [0.6, 0.6, 0.6, 1.0])
    cyl.replace_materials(mat)

    return cyl

def setup_area_lighting(loc, bg_path):
    bproc.world.set_world_background_hdr_img(bg_path)
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location(loc)
    light.set_rotation_euler([np.radians(60), 0, np.radians(45)])
    light.set_energy(250)
    
def setup_indoor_lighting(bg_path,
                          ambient_energy=5.0,
                          sun_angle=(75, -15, 0),  sun_energy=2.5,
                          key_angle=(60, -30, 0),  key_energy=35.0,
                          fill_angle=(60,  30, 0), fill_energy=12.0,
                          rim_angle=(135,  0, 0),  rim_energy=6.0):
    # 0) Background PNG (just backdrop, no emission)
    bproc.world.set_world_background_hdr_img(bg_path)

    # 1) Soft “ambient” overhead light
    amb = bproc.types.Light()
    amb.set_type("AREA")
    amb.set_rotation_euler([0, 0, 0])
    amb.set_location([0.0, 0.0, 5.0])
    amb.set_energy(ambient_energy)
    amb.set_scale([10.0, 10.0, 1.0])    # huge, very soft

    # 2) Distant SUN for directional modeling
    sun = bproc.types.Light()
    sun.set_type("SUN")
    sun.set_rotation_euler([np.radians(a) for a in sun_angle])
    sun.set_energy(sun_energy)
    sun.set_color([1.0, 0.95, 0.9])

    # 3) Key window lamp
    key = bproc.types.Light()
    key.set_type("AREA")
    key.set_rotation_euler([np.radians(a) for a in key_angle])
    key.set_location([0, -2.0, 2.5])
    key.set_energy(key_energy)
    key.set_scale([3.0, 3.0, 1.0])

    # 4) Fill lamp on opposite side
    fill = bproc.types.Light()
    fill.set_type("AREA")
    fill.set_rotation_euler([np.radians(a) for a in fill_angle])
    fill.set_location([0, 2.0, 2.5])
    fill.set_energy(fill_energy)
    fill.set_scale([2.0, 2.0, 1.0])

    # 5) Rim light for edge highlight
    rim = bproc.types.Light()
    rim.set_type("AREA")
    rim.set_rotation_euler([np.radians(a) for a in rim_angle])
    rim.set_location([2.5, 0, 2.0])
    rim.set_energy(rim_energy)
    rim.set_scale([2.0, 2.0, 1.0])



def setup_cam(euler, loc, resolution=[512,512]):
    quat    = mathutils.Euler(euler, 'XYZ').to_quaternion()
    rot_mat = np.array(quat.to_matrix())
    cam_pose = bproc.math.build_transformation_mat(loc, rot_mat)
    bproc.camera.set_resolution(*resolution)
    bproc.camera.add_camera_pose(cam_pose)

def create_box(scale, wall=0.005, name_prefix="box", tex_path=None):
    objs = []
    x, y, z = scale
    offset = 0.01

    # 1) build geometry
    bottom = bproc.object.create_primitive(
        'CUBE',
        scale=[x, y, wall],
        location=[0,0,-wall-offset]
    )
    bottom.set_name(f"{name_prefix}_bottom")
    objs.append(bottom)

    for axis, sign in [('x',1),('x',-1),('y',1),('y',-1)]:
        sx = wall if axis=='x' else x
        sy = wall if axis=='y' else (y-2*wall)
        loc = [
            (x - wall)*sign if axis=='x' else 0,
            (y - wall)*sign if axis=='y' else 0,
            z -wall-offset
        ]
        wall_obj = bproc.object.create_primitive(
            'CUBE',
            scale=[sx, sy, z],
            location=loc
        )
        wall_obj.set_name(f"{name_prefix}_wall_{axis}{'p' if sign>0 else 'n'}")
        objs.append(wall_obj)

    # 2) create & apply wood material (if provided)
    if tex_path is not None:
        wood_mat = bproc.material.create_material_from_texture(
            tex_path, material_name="box_wood"
        )
        for o in objs:
            o.replace_materials(wood_mat)
    else:
        grey = bproc.material.create(name="box_grey")
        grey.set_principled_shader_value("Base Color", [0.6, 0.6, 0.6, 1])
        for o in objs:
            o.replace_materials(grey)

    return objs


if __name__ == "__main__":
    main()
