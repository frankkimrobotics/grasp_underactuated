Grasp underactuated: Complemetary Free Manipulation to Pose Estimation Pipeline Repo

<p float="left">
  <img src="https://github.com/user-attachments/assets/2a150068-43e5-4a77-b2d6-a7978860a681" width="300" alt="fp_image" />
  <img src="https://github.com/user-attachments/assets/a9a0b723-bf3b-4e1c-b95e-7e1854bbfe24" width="300" alt="image (2)" />
</p>


This repo includes a full pipeline which can perform planning for manipulation tasks on an object using a singular end effector, generate Blender renders of each scene (along with RGB and depth images from a simulated sensor), run FoundationPose to estimate position, and then run an Unscented Kalman Filter on the final data.

As of May 2025, this repo was organized to generate a bunch of object pose estimate data based on a real manipulation trajectory so as to tune the UKF (which was tested on renders with added noise). For an online application this will have to be modified so that each module actively communicates.

Organization of the Repo:

There are four main modules, each with a separate folder: Data Collection, High Level Scripts (includes evaluation files), Manipulation (includes the UKF file), Blender Rendering, and FoundationPose State Estimation. 


1. Scripts

This has all the main files needed to run the pipeline, although currently the manipulation stack has to be run on its own (see below). rollout_imgs_main.py is the file that can run both the rendering and pose estimation modules. The eval files generate plots evaluating the estimations from either FoundationPose or the UKF output, generating a bunch of plots. The launch files are needed to setup bproc in the right env (for generating renders). add_noise_to_imgs.py adds noise to the images generated from the render (this was for tuning the UKF).


2. Complementarity-Free-Dexterous-Manipulation

This is the manipulation module, which pulls up a mujoco screen and runs one (or more) trials with a given object and end effector. The goal in each trial is to reorient said object from an initial position to a given position. Check the README inside the overall folder for more help on running examples. Currently everything in this folder has to be run in terminal.

to run an example manipulation sim, go the object folder inside the following path (Complementarity-Free-Dexterous-Manipulation/examples/mpc/singlefinger) and run one of the example/test files. These mostly save to the data folder, but check. To change the setup for each object, change the params.py file. To add a new object copy-paste one of the existing object folders, alter it, and add a new xml file to the envs folder (making sure to use it in params). To try the multi-finger examples go to the other folders in mpc. 

run_save_UKF.py is the main UKF file, which runs it on the FoundationPose data and saves a final CSV to the data folder. 

See below for generating metrics and plots from the UKF and manipulation data.


3. perception_rendering

This folder is for blender renders, using bproc. The files in scripts are the ones that generate all the renders and RGB/depth images. Objects has the objects (textures, objs...).


4. FoundationPose-main

This is the pose estimation stack, which is essentially just FoundationPose from https://nvlabs.github.io/FoundationPose/. This model takes in RGB and depth images along with an object model and then outputs 6DOF estimates of the object pose. It works great.

Currently it is used on the depth and RGB images in the data folder. It should be noted that FoundationPose essentially tracks the object, needing a mask and a quality depth and RGB image for the first frame.


5. data

This stores all the data (both temporary and final) from all the modules so that they can communicate. It is organized by overall module (estimation, manipulation, evauluation...)

