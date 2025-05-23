import subprocess, pathlib, sys
import numpy as np
VENV_PATH = "/home/frank/Desktop/Scene-Diffuser/3d"

here   = pathlib.Path(__file__).parent
script = here.parent / "perception_rendering" / "scripts" / "blender_render.py"
fp_sh  = "/home/frank/Desktop/full-sim-manip-pipeline/scripts/launch_foundationpose.sh"   # adjust if you stored it elsewhere

DEFAULT_SAVE_SUFFIX = ""

color_png   = f"/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/temp/render{DEFAULT_SAVE_SUFFIX}.png"
depth_png   = f"/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/temp/depth{DEFAULT_SAVE_SUFFIX}.png"
mask_file = f"/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/temp/mask{DEFAULT_SAVE_SUFFIX}.png"
k_txt       = "/home/frank/Desktop/full-sim-manip-pipeline/perception_rendering/camera_intrisic_matrix.txt"
results_dir = f"/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/temp_poses"

def main():
    pose1 = np.array([-0.3,0,1,1.5,0.5,0.5])
    pose2 = np.array([0.3,0.0,0.5,0,0.4,0])
    for i in range(2):
        print(f"Step {i}")
        
        # img generation
        print("Running RGB render")
        rc = render_image(pose1,pose2,mask_path = mask_file)
        if rc != 0:
            print("Failed running RGB render")
            sys.exit(rc)
        print("Finished RGB render")
        
        # pose estimation
        print("Running FoundationPose")
        rc = run_foundationpose(
            "--color_file", color_png,
            "--depth_file", depth_png,
            "--mask_file", mask_file,
            "--K_file",     k_txt,
            "--mesh_file",  "/home/frank/Desktop/full-sim-manip-pipeline/perception_rendering/objects/shampoo/shampoo/textured.obj",
            "--save_dir",   results_dir
        )
        if rc != 0:
            print("Failed running FoundationPose")
            sys.exit(rc)
        print("Finished running FoundationPose")
        
        pose1 = pose1 + np.array([0.1,0.2,-0.1,0.2,-0.1,0.1])
        pose2 = pose2 + np.array([0,0.1,0,0,0,0.1])
    sys.exit(rc)

def check_docker_permissions() -> bool:
    """
    Check if the current user is in the Docker group.
    """
    try:
        # Run the groups command and check if 'docker' is in the list of groups
        result = subprocess.run("groups", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "docker" in result.stdout:
            return True
        else:
            print("Error: You are not in the Docker group.")
            return False
    except subprocess.CalledProcessError:
        print("Error: Failed to check Docker group membership.")
        return False

def run_foundationpose(*extra_args) -> int:
    """
    Execute launch_foundationpose.sh with optional arguments.
    Returns the shell script’s exit status so callers can decide what to do.
    """
    if not check_docker_permissions():
        sys.exit(1)

    cmd = ["bash", str(fp_sh), *map(str, extra_args)]
    print("▶", " ".join(cmd))
    return subprocess.call(cmd)

def render_image(pose1,pose2,mask_path=None,save_suffix = DEFAULT_SAVE_SUFFIX) -> int:
    """
    Existing BlenderProc call (kept as-is, just moved into a function for symmetry).
    """
    # Activate the virtual environment
    activate_script = VENV_PATH + "/bin/activate_this.py"
    exec(open(activate_script).read(), {'__file__': activate_script})
    
    l11,l12,l13,r11,r12,r13 = pose1
    l21,l22,l23,r21,r22,r23 = pose2

    # Run BlenderProc with the specified arguments
    cmd = [
        "blenderproc", "run", str(script),
        "--obj1_path", "perception_rendering/objects/shampoo/shampoo/textured.obj",
        "--obj1_texture_path", "perception_rendering/objects/shampoo/shampoo/texture_map.jpg",
        "--pose1",     str(l11), str(l12), str(l13),       str(r11), str(r12), str(r13),
        "--obj2_path", "perception_rendering/objects/scissors/scissors/textured.obj",
        "--obj2_texture_path", "perception_rendering/objects/scissors/scissors/texture_map.png",
        "--pose2",     str(l21), str(l22), str(l23), str(r21), str(r22), str(r23),
        "--color_out_path",  "data/image_saves/temp/render" + save_suffix + ".png",
        "--depth_out_path",  "data/image_saves/temp/depth" + save_suffix + ".png",
        "--mask_out_path",  mask_path,
        "--K_file",  k_txt,
        "--bg_path",  "perception_rendering/random_assets/Solid_grey.png"
    ]
    print("▶", " ".join(map(str, cmd)))
    return subprocess.call(cmd)

if __name__ == "__main__":
    main()
