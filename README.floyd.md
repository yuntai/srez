## Introduction
This project is an adaptation of srez project for floydhub. You will need a floydhub account and have floyd-cli installed. See https://www.floydhub.com/welcome.

## Setup project
```bash
$ git clone https://github.com/yuntai/srez
$ cd srez
$ floyd init srez
```

## Training

The model is trained with [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The dataset is available in floydhub with dataset ID (eVDcb9LTu755t3kP2JVpK4).

```bash
floyd run --env tensorflow-1.0 --gpu --data eVDcb9LTu755t3kP2JVpK4 "python srez_main.py --run train --train_time 180"
```

After the model is trained, you will be able to check the progress of the model in the '/output/train' directory.

```bash
floyd run --env tensorflow-1.0 --data <dataset ID> "python srez_main.py --run=demo --train_dir=/input/model/sample_outputs --demo_output_dir=/output"
```

### Pre-build model
The dataset eVDcb9LTu755t3kP2JVpK4 also contains the model that was built fro 130,000,000 batches. The model is contained Tensorflow checkpoint file and can be accessed in '/input/model/...'. You can generate video file for the progression from the ...

```bash
floyd run --env tensorflow-1.0 --data <dataset ID> "python srez_main.py --run=demo --train_dir=/input/model/sample_outputs --demo_output_dir=/output"
```

You can now host this model as a REST API. This means you can send any image to this API as a HTTP request and it will be style transfered. 

### Serve mode

Floyd [run](../commands/run.md) command has a `serve` mode. This will upload the files in the current directory and run a special command - 
`python app.py`. Floyd expects this file to contain the code to run a web server and listen on port `5000`. You can see the 
[app.py](https://github.com/floydhub/fast-style-transfer/blob/master/app.py) file in the sample repository. This file handles the 
incoming request, executes the code in `evaluate.py` and returns the output.

*Note that this feature is in preview mode and is not production ready yet*

```bash
$ floyd run --env keras:py2 --data Js534T344XYBPMvWqhxJNj --mode serve
Syncing code ...
RUN ID                  NAME                              VERSION
----------------------  ------------------------------  ---------
DJSdJAVa3u7AsFEMZMBBL5  floydhub/fast-style-transfer:5          5

Path to service endpoint: https://www.floydhub.com:8000/t4AdkU6awahkT3ooNazw8c

To view logs enter:
    floyd logs DJSdJAVa3u7AsFEMZMBBL5
```

### Sending requests to the REST API

Now you can send any image file as request to this api and it will return the style transfered image.

```bash
curl -o taipei_output.jpg -F "file=@./images/taipei101.jpg" https://www.floydhub.com:8000/t4AdkU6awahkT3ooNazw8c
```

```bash
$ floyd logs <RUN_ID> -t
```

```bash
$ floyd info <RUN_UD>
```

## Changes evaluation
```bash
floyd run --env tensorflow-1.0 --data eVDcb9LTu755t3kP2JVpK4 "python srez_main.py --run=demo --train_dir=/input/model/sample_outputs --demo_output_dir=/output"
```

### play video
https://www.floydhub.com/viewer/data/g8FD7uMukrVQkKSKBSNS4o/GLynmeMR4gorYFfPrNsVHA/demo1.mp4
