from torch import nn
from RISE.evaluation import auc

from lib.func import *  
from lib.args import parse_args
from lib.dataset import parse_dataset
from lib.model import parse_model

import os
import json
import time
import tqdm
import torch
import tarfile

import numpy as np


def call_model_function(images, call_model_args=None, expected_keys=None):
    images = torch.tensor(images.transpose([0,3,1,2]), dtype=torch.float32)
    images = images.requires_grad_(True)
    target_class_idx =  call_model_args['class_idx_str']
    output = model(images.cuda())
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs
    

class Timer():
    def __init__(self):
        import time
        self.time_slot = {}
        self.additional_slot = {}
    def tik(self):
        self.start = time.time()
    def tok(self, key):
        duration = time.time() - self.start
        if key in self.time_slot:
            val = self.time_slot[key]["value"]
            it = self.time_slot[key]["it"]

            new_it = it + 1
            new_val = (val * it + duration) / new_it

            self.time_slot[key]["value"] = new_val 
            self.time_slot[key]["it"] = new_it
        else:
            self.time_slot.update({key: {}})
            self.time_slot[key]["value"] = duration 
            self.time_slot[key]["it"] = 1
    
    def adder(self, data):
        for key, v in data.items():
            if key in self.additional_slot:
                val = self.additional_slot[key]["value"]
                it = self.additional_slot[key]["it"]

                new_it = it + 1
                new_val = (val * it + v) / new_it

                self.additional_slot[key]["value"] = new_val 
                self.additional_slot[key]["it"] = new_it
            else:
                self.additional_slot.update({key: {}})
                self.additional_slot[key]["value"] = v 
                self.additional_slot[key]["it"] = 1
        
        
    def return_desc(self):
        iter_desc = ""
        for k, v in self.time_slot.items():
            iter_desc += "{}:{:3.1f}s,".format(k, v["value"])
        
        for k, v in self.additional_slot.items():
            iter_desc += "|{}:{:.0f}".format(k, v["value"])
  
        return iter_desc

import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def iteration_for_others(args, inputs,  model, insertion, deletion, timer, others):

    curr_input = inputs.cuda()
    curr_input.requires_grad_()

    pred = model(curr_input)
    prediction_index = torch.argmax(pred).tolist()

    timer.tok("model")
    timer.tik()

    target_index = prediction_index

    m_del_auc, m_ins_auc = [], []
    if args.method == "saliency":
        attribution = others.attribute(curr_input, target=target_index, abs=False)
    elif args.method == "inputxgrad":
        attribution = others.attribute(curr_input, target=target_index).detach()
    elif args.method == "integratedgrad":
        attribution = others.attribute(curr_input, target=target_index).detach()
    elif args.method == "guidedbackprop":
        attribution = others.attribute(curr_input, target=target_index).detach()
    elif args.method == "guidedgradcam":
        attribution = others.attribute(curr_input, target=target_index).detach()
    elif args.method == "blurig":
        im_orig = inputs[0].numpy().transpose([1,2,0])
        call_model_args = {'class_idx_str': prediction_index}
        im = im_orig.astype(np.float32)
        baseline = np.zeros(im.shape)
        attribution = others.GetMask(im, call_model_function, call_model_args, batch_size=20)
        attribution = np.expand_dims(attribution.transpose([2,0,1]), 0).astype(np.float32)
        attribution = torch.from_numpy(attribution)

    _del_result = deletion.single_run(inputs, attribution.cpu().numpy())
    _ins_result = insertion.single_run(inputs, attribution.cpu().numpy())

    timer.tok("pidg")
    timer.tik()

    _del_auc = auc(_del_result)
    _ins_auc = auc(_ins_result)

    m_del_auc.append(_del_auc)
    m_ins_auc.append(_ins_auc)

    return timer, m_del_auc, m_ins_auc        


def iteraion_for_ours(args, inputs, model, insertion, deletion, timer, ReLU_layer_index_list):
    
    timer.tok("data")
    timer.tik()

    inputs_cuda = inputs.to(device)
    prediction_index = torch.argmax(model(inputs_cuda)).tolist()

    path_dict = dict()

    timer.tok("model")
    timer.tik()

    prev_layer_index = None
    prev_linear_weight = None
    path_info = {}
    for layer_index in ReLU_layer_index_list[::-1]:
        max_attribute_index_list, linear_weight = \
            hidden_index_list_for_maximize_softmax(
                inputs_cuda, model, 
                layer_index=layer_index, 
                prediction_index=prediction_index, 
                prev_layer_index=prev_layer_index, 
                max_num_path=args.max_num_path)
        
        linear_weight = linear_weight.cuda()
        # Masking non-selected units
        list_of_layers = []
        for li, module in enumerate(model):
            if li == layer_index:
                list_of_layers.append(MaskLayer(max_attribute_index_list))
            else:
                list_of_layers.append(module)
        model = nn.Sequential(*list_of_layers)
    
        if (len(max_attribute_index_list) > 0):
            # units for configuration of path
            path_dict[layer_index] = max_attribute_index_list
            prev_layer_index = layer_index
            prev_linear_weight = linear_weight
        path_info.update({"L{}".format(layer_index) : len(max_attribute_index_list)})

    timer.adder(path_info)

    timer.tok("path")
    timer.tik()

    # Calculate Wp for the (in)complete path!
    W, b = linear_model_for_path_dict(inputs_cuda, model, prev_linear_weight=prev_linear_weight)

    attribution = W[:, prediction_index].view(inputs.shape) * inputs_cuda
    attribution = attribution.cpu().numpy()

    timer.tok("linM")
    timer.tik()

    _del_result = deletion.single_run(inputs, attribution)
    _ins_result = insertion.single_run(inputs, attribution)
    
    timer.tok("pidg")
    timer.tik()

    _del_auc = auc(_del_result)
    _ins_auc = auc(_ins_result)

    m_del_auc, m_ins_auc = [], []
    m_del_auc.append(_del_auc)
    m_ins_auc.append(_ins_auc)

    return timer, m_del_auc, m_ins_auc


if __name__ == "__main__":
    
    args = parse_args()

    if "debug" not in args.save_root:
        if os.path.isdir(args.save_root) is True:
            print("Check your path!")
            import pdb; pdb.set_trace()
    os.makedirs(args.save_root, exist_ok=True)
    
    tar = tarfile.open( os.path.join(args.save_root, 'sources.tar'), 'w' )
    curr_file = os.listdir(os.getcwd())
    curr_file = [i for i in curr_file if ".py" in i]
    curr_file = [tar.add(i) for i in curr_file if os.path.isdir(i) is False]
    tar.close()
    with open(os.path.join(args.save_root,'args.txt'), 'w') as f:
        json.dump(dict(vars(args)), f, indent=2)

    # For reproducibility
    random_seed = int(args.seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    torch.cuda.empty_cache()

    invTrans, test_loader, classes, \
        name2gtidx, folder_to_class, class_number, class_to_idx = parse_dataset(args)
    
    model, insertion, deletion, ReLU_layer_index_list, others = parse_model(args, class_to_idx, class_number, folder_to_class)
    

    timer = Timer()
    timer.tik()

    global_c_del_auc = []
    global_c_ins_auc = []

    global_m_del_auc = []
    global_m_ins_auc = []

    pbar = tqdm.tqdm(test_loader)
    for data_index, batch_data in enumerate(pbar):
        inputs, _, _ = batch_data

        if args.method == "ours":
            timer, m_del_auc, m_ins_auc = \
                iteraion_for_ours(args, inputs, model, insertion, deletion, timer, ReLU_layer_index_list)
        else:
            timer, m_del_auc, m_ins_auc = \
                iteration_for_others(args, inputs,  model, insertion, deletion, timer, others)
        
        timer.tok("save")
        timer.tik()

        pbar.set_description(timer.return_desc())

        global_m_del_auc.append(m_del_auc)
        global_m_ins_auc.append(m_ins_auc)


    global_m_del_auc = np.array(global_m_del_auc)
    global_m_ins_auc = np.array(global_m_ins_auc)

    insertion_result_desc = "Insertion Game (Higher is better):\n\tMerge: {:5.3f}".format(np.mean(global_m_ins_auc))
    deletion_result_desc = "Deletion Game (Lower is better):\n\tMerge: {:5.3f}".format(np.mean(global_m_del_auc))

    
    buffernum = timer.return_desc().split(",")

    unitnum = []
    unitnum.extend(buffernum[:-1])
    unitnum.extend(buffernum[-1].split("|")[1:])

    np.savetxt(os.path.join(args.save_root, "result.txt"), [insertion_result_desc, deletion_result_desc], fmt="%s")
    np.savetxt(os.path.join(args.save_root, "avg_unit_num.txt"), unitnum, fmt="%s")
    
    print(insertion_result_desc)
    print(deletion_result_desc)