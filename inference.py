import argparse
import models


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the TTCDehazeNet inference")

    parser.add_argument('--gt_folder', type=str, help='Path to the GT folder or image', default=r'F:\Research\HazeSpace2M\data\Fog\Haze\OOTSEHL1_1.jpg', required=False)
    parser.add_argument('--hazy_folder', type=str, help='Path to the hazy folder or image', default=r'F:\Research\HazeSpace2M\data\Fog\Haze\OOTSEHL1_3.jpg', required=False)
    parser.add_argument('--output_dir', type=str, help='Directory to save dehazed images', default= r"./storage/Test1/", required=False)

    parser.add_argument('--classifier', type=str, help='Path to the classifier model', default= "./pretrained_weights/classifiers/ResNet152.pth", required=False)
    parser.add_argument('--cloudSD', type=str, help='Path to the cloud dehazer model', default= r"./pretrained_weights/dehazers/LD_Net_Cloud.pth", required=False)
    parser.add_argument('--ehSD', type=str, help='Path to the EH dehazer model', default= r"./pretrained_weights/dehazers/LD_Net_EH.pth", required=False)
    parser.add_argument('--fogSD', type=str, help='Path to the fog dehazer model', default= r"./pretrained_weights/dehazers/LD_Net_Fog.pth", required=False)

    return parser.parse_args()


def main():
    args = parse_arguments()

    dehazers = [args.cloudSD, args.ehSD, args.fogSD]

    models.TTCDehazeNet(gt_image= args.gt_folder, hazy_image=args.hazy_folder, dehazers=dehazers, classifier= args.classifier,
                        output_dir=args.output_dir)


if __name__ == "__main__":
    main()

gt_folder = r"F:\Research\HazeSpace2M\data\Fog\Haze\OOTSEHL1_1.jpg"
hazy_folder = r"F:\Research\HazeSpace2M\data\data\haze"


output_dir = r"F:\Research\HazeSpace2M\storage\Dehazed"

classifier = r"F:\Research\HazeSpace2M\pretrained_weights\classifiers\ResNet152.pth"
cloudSD = r"F:\Research\HazeSpace2M\pretrained_weights\dehazers\LD_Net_Cloud.pth"
ehSD = r"F:\Research\HazeSpace2M\pretrained_weights\dehazers\LD_Net_EH.pth"
fogSD = r"F:\Research\HazeSpace2M\pretrained_weights\dehazers\LD_Net_Fog.pth"