import argparse
from collections import OrderedDict

import torch

def merge(args):
    """
    merge different parts of the checkpoint to the end file
    :param args:
    :return:
    """
    checkpoint0 = torch.load(args.in_file0)
    checkpoint1 = torch.load(args.in_file1)
    in_state_dict0 = checkpoint0.pop('state_dict')
    in_state_dict1 = checkpoint1.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict0.items():
        # only copy the decomposition weight from the checkpoint0
        if 'rgb_completion' in key:
            continue
        else:
            out_state_dict[key] = val

    for key, val in in_state_dict1.items():
        # only copy the completion weight from the chectkpoint1
        if 'rgb_completion' in key and 'net_D_scene' not in key:
            out_state_dict[key] = val
        else:
            continue

    checkpoint0['state_dict'] = out_state_dict
    torch.save(checkpoint0, args.out_file)


def main():
    parser = argparse.ArgumentParser(description='merge model from different files')
    parser.add_argument('--in_file0', type=str, default='./work_dirs', help='input checkpoint file (decomposition)')
    parser.add_argument('--in_file1', type=str, default='./work_dirs', help='input checkpoint file (completion)')
    parser.add_argument('--out_file', type=str, default='./work_dirs', help='output checkpoint file (end)')
    args = parser.parse_args()

    merge(args)


if __name__ == '__main__':
    main()