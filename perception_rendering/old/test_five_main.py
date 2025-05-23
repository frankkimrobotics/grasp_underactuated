# launch_five.py  (or whatever you named it)
import subprocess, pathlib

ROOT   = pathlib.Path(__file__).parent            # folder that holds the assets
SCRIPT = ROOT / "five_main.py"
BPROC  = "blenderproc"

# cases = [
#     {"obj": ROOT / "assets/can.obj",
#      "tex": ROOT / "assets/can.jpg"},
# ]

# for i, c in enumerate(cases):
out_dir = "/home/frank/Desktop/sim_pipeline_perception_manipulation/image_saves"
cmd = [
    BPROC, "run", str(SCRIPT),
    "--obj_path",  "/home/frank/Desktop/sim_pipeline_perception_manipulation/objects/shampoo/shampoo/collision.obj",
    "--tex_path",  "/home/frank/Desktop/sim_pipeline_perception_manipulation/objects/shampoo/shampoo/texture_map.jpg",
    "--box_scale", "1.3", "1.5", "0.5",
    "--obj_scale", "3", "3", "3",
    "--output_dir", "/home/frank/Desktop/sim_pipeline_perception_manipulation/image_saves"
]
subprocess.run(cmd, check=True)
