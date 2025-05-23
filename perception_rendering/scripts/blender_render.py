import blenderproc as bproc          # ← must be first!
import argparse
import imageio
import numpy as np
import mathutils
from pathlib import Path

def parse_pose(vals):
    if len(vals) != 6:
        raise argparse.ArgumentTypeError("pose needs 6 floats")
    v = list(map(float, vals))
    return np.array(v[:3]), np.array(v[3:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj1_path", required=True)
    ap.add_argument("--obj1_texture_path", required=True)
    ap.add_argument("--pose1",     nargs=6, required=True)
    ap.add_argument("--obj1_scale",  nargs=3, type=float, default=[5,5,5])
    ap.add_argument("--obj2_path", required=True)
    ap.add_argument("--obj2_texture_path", required=True)
    ap.add_argument("--pose2",     nargs=6, required=True)
    ap.add_argument("--obj2_scale",  nargs=3, type=float, default=[5,5,5])
    ap.add_argument("--box_scale",  nargs=3, type=float, default=[1.3,1.5,0.25])
    ap.add_argument("--color_out_path", required=True)
    ap.add_argument("--depth_out_path", default=None,
                help="Optional: write depth PNG (16-bit mm)")
    ap.add_argument("--mask_out_path", default=None)
    ap.add_argument("--bg_path", required=True)
    ap.add_argument("--K_file", required=True)
    args = ap.parse_args()

    loc1, rot1 = parse_pose(args.pose1)
    loc2, rot2 = parse_pose(args.pose2)

    # --------------------------------------------------------------------------
    bproc.init()
    boxes = create_box(args.box_scale)
    for b in boxes:
        b.set_cp("category_id", 0)
    obj1 = load_obj_with_texture(args.obj1_path, rot1,loc1, texture_path=args.obj1_texture_path, scale=args.obj1_scale)
    obj1.set_cp("category_id", 1)
    obj2 = load_obj_with_texture(args.obj2_path, rot2,loc2, texture_path=args.obj2_texture_path, scale=args.obj2_scale)
    obj2.set_cp("category_id", 2)
        
    cam_euler = (np.radians(40), 0, 0)
    cam_loc = [0, -2.5, 3]
    setup_cam(cam_euler,cam_loc)
    W, H = 512, 512  

    lighting_loc = [0, 0, 5]
    setup_area_lighting(lighting_loc)
    K = np.loadtxt(args.K_file, dtype=np.float32)
    bproc.camera.set_intrinsics_from_K_matrix(K, W, H)

    if args.depth_out_path:
        depth_out = Path(args.depth_out_path)
        depth_out.parent.mkdir(parents=True, exist_ok=True)
        # enable depth-output to a temporary .exr for conversion
        bproc.renderer.enable_depth_output(
            activate_antialiasing=True,
            output_dir=str(depth_out.parent),
            file_prefix="depth_",
            output_key="depth",
            antialiasing_distance_max=10.0,
            convert_to_distance=True,
        )
    print(bproc.python.types.StructUtilityFunctions.get_instances())
    data = bproc.renderer.render()
    rgb   = data["colors"][0]
    depth = data["depth"][0]
    
    
    imageio.imwrite(args.color_out_path, rgb)
    print(f"saved color img → {args.color_out_path}")

    # depth (in metres): assumes 'depth' key is present
    if args.depth_out_path:
        # convert to 16-bit PNG in millimetres
        depth_mm = (depth * 1000.0).astype(np.uint16)
        imageio.imwrite(args.depth_out_path, depth_mm)
        print(f"saved depth img → {args.depth_out_path}")

    
    if args.mask_out_path:
        seg = bproc.renderer.render_segmap(map_by=["instance","name"])
        inst_map = seg["instance_segmaps"][0]
        attr    = seg["instance_attribute_maps"][0]
        name2idx = {m["name"]:m["idx"] for m in attr}
        my_idx   = name2idx[obj1.get_name()]
        mask    = (inst_map == my_idx).astype(np.uint8) * 255
        Path(args.mask_out_path).parent.mkdir(exist_ok=True, parents=True)
        imageio.imwrite(args.mask_out_path, mask)
        print(f"saved mask → {args.mask_out_path}")
        
        
def load_obj_with_texture(obj_path, rot, loc, texture_path=None, scale=None):
    """
    Loads an OBJ, places/scales it, and assigns either:
      • the original OBJ materials (if texture_path is None) or
      • a fresh Principled BSDF with `texture_path` on Base-Color.

    Returns the BlenderProc object.
    """
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_location(loc)
    obj.set_rotation_euler(rot)
    if scale is not None:
        obj.set_scale(scale)

    if texture_path is not None:
        mat = bproc.material.create_material_from_texture(texture_path,
                                                          material_name="tex_mat")
        obj.replace_materials(mat)

    return obj

def setup_area_lighting(loc):
    # lighting: bright area light + dim neutral environment
    bproc.world.set_world_background_hdr_img("/home/frank/Desktop/full-sim-manip-pipeline/perception_rendering/random_assets/Solid_grey.png")    #  bg
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location(loc)
    light.set_rotation_euler([np.radians(60), 0, np.radians(45)])
    light.set_energy(1000)

def setup_cam(euler,loc,resolution = [512, 512]):
    quat = mathutils.Euler(euler, 'XYZ').to_quaternion()
    rot_mat3 = np.array(quat.to_matrix())
    cam_pose = bproc.math.build_transformation_mat(
    loc,                  # xyz
    rot_mat3  
    )
    bproc.camera.set_resolution(resolution[0], resolution[1])
    bproc.camera.add_camera_pose(cam_pose)

def create_box(scale, wall=0.02, name_prefix="box"):
    """
    Build an open container centred at (0,0,z_scale) with wall-thickness `wall`.
    Returns a list with the 5 BlenderProc objects (4 walls + bottom).
    """
    objs = []
    x_scale, y_scale, z_scale = scale
    # bottom
    bottom = bproc.object.create_primitive(
        'CUBE',
        scale=[x_scale, y_scale, wall],
        location=[0, 0, wall])            # rests on Z=0
    bottom.set_name(f"{name_prefix}_bottom")
    objs.append(bottom)

    # +X wall
    wall_px = bproc.object.create_primitive(
        'CUBE',
        scale=[wall, y_scale, z_scale],
        location=[x_scale - wall, 0, z_scale + wall])
    wall_px.set_name(f"{name_prefix}_wall_px")
    objs.append(wall_px)

    # –X wall
    wall_nx = bproc.object.create_primitive(
        'CUBE',
        scale=[wall, y_scale, z_scale],
        location=[-x_scale + wall, 0, z_scale + wall])
    wall_nx.set_name(f"{name_prefix}_wall_nx")
    objs.append(wall_nx)

    # +Y wall
    wall_py = bproc.object.create_primitive(
        'CUBE',
        scale=[x_scale, wall, z_scale],
        location=[0, y_scale - wall, z_scale + wall])
    wall_py.set_name(f"{name_prefix}_wall_py")
    objs.append(wall_py)

    # –Y wall
    wall_ny = bproc.object.create_primitive(
        'CUBE',
        scale=[x_scale, wall, z_scale],
        location=[0, -y_scale + wall, z_scale + wall])
    wall_ny.set_name(f"{name_prefix}_wall_ny")
    objs.append(wall_ny)

    # Optional: give them a light-grey material
    grey = bproc.material.create(name="box_grey")
    grey.set_principled_shader_value("Base Color", [0.6, 0.6, 0.6, 1])
    for o in objs:
        o.replace_materials(grey)

    return objs



if __name__ == "__main__":
    main()
