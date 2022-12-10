from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

import argparse
import sys
import os


# main process
def main(args):
    image_path = args.image_path
    threshold = args.threshold

    # prepare models
    mtcnn = MTCNN(image_size = 160, margin = 0)
    resnet = InceptionResnetV1(pretrained = 'vggface2').eval()

    # prepare lists
    img_embeddings = []
    img_names = []

    # scan all images
    img_cnt = 0
    img_list = os.listdir(image_path)
    for img_file in img_list:
        # skip unnecessary files
        if img_file == '.DS_Store':
            continue
        if img_file == 'thumbs.db':
            continue

        # calculate embeddings
        file_path = image_path + '\\' + img_file
        img_org = Image.open(file_path)
        img_cropped = mtcnn(img_org)
        img_embedding = resnet(img_cropped.unsqueeze(0))

        # insert to list
        img_embeddings.append(img_embedding)
        img_names.append(img_file)
        img_cnt += 1

        # free space
        del img_embedding
        del img_cropped
        del img_org

    # check similarity
    dists = [[(emb1 - emb2).norm().item() for emb2 in img_embeddings] for emb1 in img_embeddings]
    for i1 in range(img_cnt):
        for i2 in range(i1 + 1, img_cnt):
            if (dists[i1][i2] < threshold) and (i1 != i2):
                print(img_names[i1] + ' : ' + img_names[i2] + ' = ' + str(dists[i1][i2]))


# argument parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # image size information
    parser.add_argument('--image_path', type = str,
        help = 'Image path', default = './images')
    parser.add_argument('--threshold', type = float,
        help = 'Threshold for similarity', default = 0.5)

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

