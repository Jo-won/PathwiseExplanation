from torch import nn
from lib.jacobian import jacobian

import torch

class MaskLayer(nn.Module):
    def __init__(self, mask_index):
        super(MaskLayer, self).__init__()
        self.mask_index = mask_index

    def forward(self, x):
        n_unit = x.view(1, -1).shape[-1]
        mask = torch.ones(n_unit)
        mask[self.mask_index] = 0
        mask = mask.bool()
        x_c = x.view(1, -1).clone()
        x_c[:, mask] = 0
        return x_c.view_as(x)


def hidden_index_list_for_maximize_softmax(inputs, model, layer_index, prediction_index, prev_layer_index=None, 
                                           max_num_path="-1", return_unitnum=False):    
    
    n_classes = model[-1].out_features
    d_prev_h_d_h = None 

    if (prev_layer_index != None):
        lower_func = lambda x: model[:(layer_index+1)](x)
        higher_func = lambda x: model[(layer_index+1):](x)

        h = lower_func(inputs)
        d_y_d_h_via_prev_h = jacobian(higher_func, h)
        d_y_d_h_via_prev_h = d_y_d_h_via_prev_h.view(n_classes, -1).transpose(1, 0)  # Note! This is for one-batch case!

        h_view = h.view(1, -1)
        Wx_list = h_view[0][:, None] * d_y_d_h_via_prev_h
        
        softmax_Wx_list = torch.nn.functional.softmax(Wx_list, dim=1)[:, prediction_index]


        sorted_attribute_index_list = torch.sort(softmax_Wx_list, descending=True)[1]
        if (max_num_path != "-1"):
            if "%" not in max_num_path:
                sorted_attribute_index_list = sorted_attribute_index_list[:int(max_num_path)]
            else:
                num = max(1, round(len(sorted_attribute_index_list) / 100 * float(max_num_path.split("%")[0])))
                sorted_attribute_index_list = sorted_attribute_index_list[:num]

        sorted_softmax_Wx_list = softmax_Wx_list[sorted_attribute_index_list]
        filtered_idx = sorted_softmax_Wx_list > (1. / n_classes)
        if torch.sum(filtered_idx).item() == 0:
            max_ind = torch.argmax(sorted_softmax_Wx_list).item()
            filtered_idx[max_ind] = True
        max_attribute_index_list = sorted_attribute_index_list[filtered_idx].tolist()
        if len(max_attribute_index_list) == 0:
            import pdb; pdb.set_trace()
       

        max_attribute_index_list = sorted(max_attribute_index_list)
        linear_weight = d_y_d_h_via_prev_h[torch.tensor(max_attribute_index_list)]
        
        if return_unitnum:
            del h_view, d_prev_h_d_h, d_y_d_h_via_prev_h, Wx_list
        else:
            del h_view, d_prev_h_d_h, d_y_d_h_via_prev_h, Wx_list, softmax_Wx_list
    else:
        lower_func = lambda x: model[:(layer_index+1)](x)
        higher_func = lambda x: model[(layer_index+1):](x)

        h = lower_func(inputs)
        
        d_y_d_h = jacobian(higher_func, h)
        d_y_d_h = d_y_d_h.view(n_classes, -1).transpose(0, 1)

        h = h.view(1, -1)
        Wx_list = h[0][:, None] * d_y_d_h
       

        softmax_Wx_list = torch.nn.functional.softmax(Wx_list, dim=1)[:, prediction_index]
        sorted_attribute_index_list = torch.sort(softmax_Wx_list, descending=True)[1]
        if (max_num_path != "-1"):
            if "%" not in max_num_path:
                sorted_attribute_index_list = sorted_attribute_index_list[:int(max_num_path)]
            else:
                num = max(1, round(len(sorted_attribute_index_list) / 100 * float(max_num_path.split("%")[0])))
                sorted_attribute_index_list = sorted_attribute_index_list[:num]


        sorted_softmax_Wx_list = softmax_Wx_list[sorted_attribute_index_list]
        filtered_idx = sorted_softmax_Wx_list > (1. / n_classes)
        if torch.sum(filtered_idx).item() == 0:
            max_ind = torch.argmax(sorted_softmax_Wx_list).item()
            filtered_idx[max_ind] = True
        max_attribute_index_list = sorted_attribute_index_list[filtered_idx].tolist()
        if len(max_attribute_index_list) == 0:
            import pdb; pdb.set_trace()


        max_attribute_index_list = sorted(max_attribute_index_list)
        linear_weight = d_y_d_h[torch.tensor(max_attribute_index_list)]
        if return_unitnum:
            del h, d_y_d_h, Wx_list
        else:
            del h, d_y_d_h, Wx_list, softmax_Wx_list
   

    if return_unitnum:
        return max_attribute_index_list, linear_weight, len(softmax_Wx_list)
    else:
        return max_attribute_index_list, linear_weight

def linear_model_for_path_dict(inputs, model, prev_linear_weight=None):
    n_classes = model[-1].out_features
    
    linear_weight = None
    linear_bias = None
    if (linear_weight != None) | (prev_linear_weight == None):
        import pdb; pdb.set_trace()
    linear_weight = jacobian(model, inputs)
    linear_weight = linear_weight.view(n_classes, -1).transpose(1, 0)
    return linear_weight, linear_bias 
    
