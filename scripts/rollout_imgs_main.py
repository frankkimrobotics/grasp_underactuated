import subprocess, pathlib, sys
import numpy as np
VENV_PATH = "/home/frank/Desktop/Scene-Diffuser/3d"

here   = pathlib.Path(__file__).parent
script = here.parent / "perception_rendering" / "scripts" / "blender_rollout_render.py"
fp_sh  = "/home/frank/Desktop/full-sim-manip-pipeline/scripts/launch_rollout_foundationpose.sh"   # adjust if you stored it elsewhere

DEFAULT_SAVE_SUFFIX = ""

color_dir   = "/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard/rgb"
depth_dir   = "/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard/depth"
mask_dir = "/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard/masks"
pose_file = "/home/frank/Desktop/full-sim-manip-pipeline/data/CFDM_poses/true_poses/mustard/final_testing_poses.csv"
k_txt       = "/home/frank/Desktop/full-sim-manip-pipeline/perception_rendering/camera_intrisic_matrix.txt"
results_dir = "/home/frank/Desktop/full-sim-manip-pipeline/data/FoundationPose_estimations/mustard"


def main():
    
    # # # # img generation
    # print("Running RGB render")
    # rc = render_image(color_dir,depth_dir,mask_dir,pose_file)
    # if rc != 0:
    #     print("Failed running RGB render")
    #     sys.exit(rc)
    # print("Finished RGB render")
    
    # # # pose estimation
    print("Running FoundationPose")
    rc = run_foundationpose(
        "--test_scene_dir", "/home/frank/Desktop/full-sim-manip-pipeline/data/image_saves/mustard_testing_final_w_noise",
        # "--K_file",     k_txt,
        "--save_dir",   results_dir,
        "--mesh_file",  "/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/demo_data/mustard0/mesh/textured_simple.obj"
    )
    if rc != 0:
        print("Failed running FoundationPose")
        sys.exit(rc)
    print("Finished running FoundationPose")
    
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

def render_image(color_dir,depth_dir,mask_dir,pose_file) -> int:
    """
    Existing BlenderProc call (kept as-is, just moved into a function for symmetry).
    """
    # Activate the virtual environment
    activate_script = VENV_PATH + "/bin/activate_this.py"
    exec(open(activate_script).read(), {'__file__': activate_script})

    # Run BlenderProc with the specified arguments
    cmd = [
        "blenderproc", "run", str(script),
        "--obj_path", "/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/demo_data/mustard0/mesh/textured_simple.obj",
        "--obj_texture_path", "/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/demo_data/mustard0/mesh/texture_map.png",
        "--obj_texture_mtl_path", "/home/frank/Desktop/full-sim-manip-pipeline/FoundationPose-main/demo_data/mustard0/mesh/textured_simple.obj.mtl",
        "--pose_csv",  pose_file,
        "--color_out_dir",  color_dir,
        "--depth_out_dir",  depth_dir,
        "--mask_out_dir",  mask_dir,
        "--K_file",  k_txt,
        "--bg_path",  "perception_rendering/random_assets/dark_brown.jpg",
        "--box_tex_path",  "perception_rendering/random_assets/box_tex.jpg"
    ]
    print("▶", " ".join(map(str, cmd)))
    return subprocess.call(cmd)

if __name__ == "__main__":
    main()
