import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='gpu', default=6)
    parser.add_argument('--seed', help='seed', default=0)
    parser.add_argument('--save_root', help='save_root', default="jobs/pidg_data")
    parser.add_argument('--dataset', help="imagenet_select or cifar10", default="imagenet_select")
    parser.add_argument('--method', help='Choose target method', default="ours", 
                        choices=['ours', 'saliency', 'inputxgrad', 'integratedgrad', 'guidedbackprop', 'guidedgradcam', 'inputxguidedbackprop', 'blurig'])
    
    parser.add_argument('--max_num_path', help='max_num_path (width)', default="-1")
    parser.add_argument('--max_index_path', help='max_index_path (last layer - depth)', default=-1, type=int)
    parser.add_argument('--model_type', help='Choose target method', default="vgg16")


    args = parser.parse_args()

    if isinstance(args.max_num_path, int):
        args.max_num_path = str(args.max_num_path)

    if args.method == "ours":
        args.save_root += "_n{}_m{}".format(
            args.max_index_path if args.max_index_path!=-1 else "All", 
            args.max_num_path if args.max_num_path!="-1" else "All")
    if (args.model_type != "toyv1") & ("imagenet" not in args.dataset):
        args.save_root += "_{}".format(args.model_type)


    
        
    return args