OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/imgNS_others_saliency \
    --dataset imagenet_select \
    --method saliency

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/imgNS_others_inputxgrad \
    --dataset imagenet_select \
    --method inputxgrad

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/imgNS_others_integratedgrad \
    --dataset imagenet_select \
    --method integratedgrad

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/imgNS_others_guidedbackprop \
    --dataset imagenet_select \
    --method guidedbackprop

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/imgNS_others_guidedgradcam \
    --dataset imagenet_select \
    --method guidedgradcam
