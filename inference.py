import argparse
import models


def parse_arguments():
    """
    Parses the command-line arguments provided by the user.

    ## To run this script from the command line, you can provide arguments like so:
    python inference.py --gt_folder <path_to_gt>
    --hazy_folder <path_to_hazy> --output_dir <output_dir>
    --classifier <path_to_classifier> --cloudSD <path_to_cloudSD>
    --ehSD <path_to_ehSD> --fogSD <path_to_fogSD>

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run the HazeSpace2M inference")

    parser.add_argument('--gt_folder', type=str,
                        help='Path to the Ground Truth (GT) folder or image.',
                        default=r'F:\Research\HazeSpace2M\data\Fog\Haze\OOTSEHL1_1.jpg',
                        required=False)

    parser.add_argument('--hazy_folder', type=str,
                        help='Path to the hazy folder or image. It can handle both single image and folder of images.',
                        default=r'F:\Research\HazeSpace2M\data\Fog\Haze\OOTSEHL1_3.jpg',
                        required=False)

    parser.add_argument('--output_dir', type=str,
                        help='Directory to save dehazed images.',
                        default=r"./storage/Test1/",
                        required=False)

    parser.add_argument('--classifier', type=str,
                        help='Path to the classifier model. The classifier you want to use for predicting the haze type and conditional dehazing.',
                        default="./pretrained_weights/classifiers/ResNet152.pth",
                        required=False)

    parser.add_argument('--cloudSD', type=str,
                        help='Path to the cloud specialized dehazer model.',
                        default=r"./pretrained_weights/dehazers/LD_Net_Cloud.pth",
                        required=False)

    parser.add_argument('--ehSD', type=str,
                        help='PPath to the EH specialized dehazer model.',
                        default=r"./pretrained_weights/dehazers/LD_Net_EH.pth",
                        required=False)

    parser.add_argument('--fogSD', type=str,
                        help='Path to the Fog specialized dehazer model.',
                        default=r"./pretrained_weights/dehazers/LD_Net_Fog.pth",
                        required=False)

    return parser.parse_args()



def main():
    """
    Main function to run the HazeSpace2M inference based on provided arguments.
    """
    args = parse_arguments()

    dehazers = [args.cloudSD, args.ehSD, args.fogSD]

    models.conditionalDehazing(gt_image=args.gt_folder, hazy_image=args.hazy_folder, dehazers=dehazers,
                        classifier=args.classifier, output_dir=args.output_dir)


if __name__ == "__main__":
    main()



