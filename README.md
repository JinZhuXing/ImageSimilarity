# Image Similarity

This project is for finding similarity images in the directory.

## Prepare

This project was tested in Python 3.7.

You have install python packages with following command:

    $ pip install -r requirements.txt

After that, you have to copy [pre-trained model](https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt) to cache directory ({User Directory}/.cache/torch/checkpoints/).

## Usage

You can run this project with following command:

    $ python imgsim.py image_path='./images' threshold=0.5

Arguments are like this:

| Argument Name | Description | Default Value |
| :---: | :---: | :---: |
| image_path | Path for images | ./images |
| threshold | Threshold for similarity | 0.5 |
