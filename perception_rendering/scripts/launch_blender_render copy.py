# launch_blander_render.py
import subprocess, pathlib, sys

here   = pathlib.Path(__file__).parent
script = here / "blender_render.py"

cmd = [
    "blenderproc", "run", str(script),
    "--obj1_path", "/home/frank/Desktop/full-sim-manip-pipeline/perception_rendering/objects/shampoo/shampoo/textured.obj",
    "--obj1_texture_path", "perception_rendering/objects/shampoo/shampoo/texture_map.jpg",
    "--pose1",     "-0.3", "0", "1",       "1.5", "0.5", "0.5",
    "--obj2_path", "perception_rendering/objects/scissors/scissors/textured.obj",
    "--obj2_texture_path", "perception_rendering/objects/scissors/scissors/texture_map.png",
    "--pose2",     "0.3", "0.0", "0.1", "0", "0.4", "0",
    "--out_path",  "perception_rendering/image_saves/render.png"
]

# forward BlenderProc’s exit status to the shell
sys.exit(subprocess.call(cmd))
