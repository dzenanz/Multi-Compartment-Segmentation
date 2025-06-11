import os
import shlex
import sys

sys.path.append('..')
from IterativePredict_1X import predict
from segmentation_school import get_argument_parser


def predict_all(model_file, in_directory, out_directory):
    """
    Run prediction on all WSIs in the directory using the given model.
    """
    ok = False
    arg_string = f'--base_dir "{out_directory}" --modelfile "{model_file}"'
    for filename in os.listdir(in_directory):
        if filename.endswith(".svs"):
            outfile = os.path.join(out_directory, filename[:-4] + ".xml")
            if os.path.exists(outfile):
                print(f"Skipping {filename}, output exists: {outfile}")
            else:
                file_path = os.path.join(in_directory, filename)
                arg_string += f' --files "{file_path}"'
                ok = True

    if ok:
        parser = get_argument_parser()
        args = parser.parse_args(shlex.split(arg_string))
        predict(args)
    else:
        print(f"No .svs files found in {in_directory}.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_dir.py <final_model.pth> <WSI_directory> [output_XML_directory]")
        exit(1)
    elif len(sys.argv) == 3:
        predict_all(sys.argv[1], sys.argv[2], sys.argv[2])
    else:
        predict_all(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
