Large-scale CelebFaces Attributes (CelebA) Dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

download link:
https://drive.google.com/uc?export=download&?id=0B7EVK8r0v71pZjFTYXZWM3FlRnMbuild.tar.xz

floyd run --env tensorflow-1.0 --gpu --data eVDcb9LTu755t3kP2JVpK4 "python srez_main.py --run train --train_time 180"

floyd run --env tensorflow-1.0 --data eVDcb9LTu755t3kP2JVpK4 "python srez_main.py --run=demo --train_dir=/input/model/sample_outputs --demo_output_dir=/output"
