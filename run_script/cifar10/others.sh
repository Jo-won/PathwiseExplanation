OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/cifar10_others_saliency \
    --dataset cifar10 \
    --model_type "toyv1" \
    --method saliency \
    --gpu 4

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/cifar10_others_inputxgrad \
    --dataset cifar10 \
    --model_type "toyv1" \
    --method inputxgrad \
    --gpu 4

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/cifar10_others_integratedgrad \
    --dataset cifar10 \
    --model_type "toyv1" \
    --method integratedgrad \
    --gpu 4

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/cifar10_others_guidedbackprop \
    --dataset cifar10 \
    --model_type "toyv1" \
    --method guidedbackprop \
    --gpu 4

OMP_NUM_THREADS=1 python main.py \
    --save_root jobs/cifar10_others_guidedgradcam \
    --dataset cifar10 \
    --model_type "toyv1" \
    --method guidedgradcam \
    --gpu 4
