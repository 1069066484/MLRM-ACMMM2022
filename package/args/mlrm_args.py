import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0) # round
    parser.add_argument('--seed', type=int, default=0) # savet
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--savet', type=int, default=10)
    parser.add_argument('--l2norm', type=int, default=0)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--individual_trp', type=int, default=1)
    parser.add_argument('--ovl', type=int, default=0)
    parser.add_argument('--vanilla', type=int, default=0)
    parser.add_argument('--mapmin', type=float, default=0)
    parser.add_argument('--cp', type=int, default=0)
    parser.add_argument('--single', type=int, default=1)
    parser.add_argument('--only_inter_layer', type=int, default=0)
    parser.add_argument('--lama_sig', type=int, default=3)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--bb', type=str, default='dn169', help='[res50] or [dn169] or [ic3] or [se50] or [cse50]')
    parser.add_argument('--dataset', type=str, default='chairs2')
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--base_debug', type=int, default=0)
    parser.add_argument('--tf_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--trp', type=float, default=0.3)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')
    parser.add_argument('--auto_stop', type=int, default=9999999)
    parser.add_argument('--max_step', type=int, default=150000)
    parser.add_argument('--folder_top', type=str, default='')
    parser.add_argument('--decay', type=float, default=0.1)
    parser.add_argument('--atts', type=int, default=3)
    parser.add_argument('--level_start', type=int, default=1)
    parser.add_argument('--aug_params', type=str, default="{'degrees': 20, 'scale':0.1, 'shear': 5, 'flip': 2, 'crop': 1.15, 'rgb': 0.6}")
    parser.add_argument('--weights', type=str)
    return parser.parse_args()





