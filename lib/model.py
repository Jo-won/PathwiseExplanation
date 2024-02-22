import torch
import torchvision 

from torch import nn
from RISE.evaluation import CausalMetric, auc
from scipy.ndimage import gaussian_filter
import numpy as np

from captum.attr import IntegratedGradients, InputXGradient, GuidedBackprop, Saliency, GuidedGradCam
import pytorch_cifar.models as pyci


def gkern(klen=11, nsig=5, n_channel=3):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((n_channel, n_channel, klen, klen))
    for channel_index in range(n_channel):
        kern[channel_index, channel_index] = k
    return torch.from_numpy(kern.astype('float32'))


def parse_model(args, class_to_idx, class_number, folder_to_class):

    if args.dataset == "cifar10":
        if args.model_type == "toyv1":
            model = pyci.ToyModelv1()
            model.load_state_dict(torch.load("pytorch_cifar/toycnnv1_checkpoint/ckpt.pth")['net'])
    else:
        model = torchvision.models.vgg16(pretrained=True)
        if args.dataset == "imagenet_select":
            model.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=256, out_features=64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=10)
            )
            model.load_state_dict(torch.load('./data/ILVCRL_10.pt'))
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    ReLU_layer_index_list = []
    if args.method == "ours":
        # # for simplicity of pathwise implemenation
        # Change from
        # Model(
        # "features": [0] nn.Conv, [1] nn.ReLU, ... [10] nn.ReLU
        # "classifier": [0] nn.Linear, ...
        # )
        # to
        # Model{
        # [0] nn.Conv, [1] nn.ReLU, ... [10] nn.ReLU, [11] nn.Linear, ...
        # }
        list_of_layers = [module for module in model.features.modules() if not isinstance(module, nn.Sequential)]
        if hasattr(model, "avgpool"): list_of_layers.extend([module for module in model.avgpool.modules() if not isinstance(module, nn.Sequential)]);
        list_of_layers.append(nn.Flatten())
        list_of_layers.extend([module for module in model.classifier.modules() if not isinstance(module, nn.Sequential)])
        model = nn.Sequential(*list_of_layers)

        # Find ReLU index in a model for our pathwise implemenation!
        for layer_index, layer in enumerate(model):
            if isinstance(layer, torch.nn.ReLU):
                ReLU_layer_index_list.append(layer_index)

        if int(args.max_index_path) != -1:
            ReLU_layer_index_list = ReLU_layer_index_list[-int(args.max_index_path):]

    model.cuda()
    model.eval()
    print(model)
    torch.cuda.empty_cache()


    # This is for CausalMetric (Insertion & Deletion)
    model_softmax = nn.Sequential(model, nn.Softmax(dim=1))
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig, n_channel=3)
    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)

    if args.dataset=="cifar10": step = 32;
    elif "imagenet" in args.dataset: step = 224;
    else: import pdb; pdb.set_trace()

    insertion = CausalMetric(model_softmax, 'ins', step, substrate_fn=blur)
    deletion = CausalMetric(model_softmax, 'del', step, substrate_fn=torch.zeros_like)


    print("Max Width: {}".format(args.max_num_path if args.max_num_path != "-1" else "Total!"))
    print("Max Depth: {}".format(args.max_index_path if args.max_index_path != "-1" else "Total!"))
    if len(ReLU_layer_index_list)!=0:
        print("Target Layer: {}".format(ReLU_layer_index_list))



    if args.method == "saliency":
        others = Saliency(model)
    elif args.method == "inputxgrad":
        others = InputXGradient(model)
    elif args.method == "integratedgrad":
        others = IntegratedGradients(model)
    elif args.method == "guidedbackprop":
        others = GuidedBackprop(model)
    elif args.method == "guidedgradcam":
        others = GuidedGradCam(model, model.features[-1])
    elif args.method == "blurig":
        others = saliency.BlurIG()
    else:
        others = None
        
    return model, insertion, deletion, ReLU_layer_index_list, others