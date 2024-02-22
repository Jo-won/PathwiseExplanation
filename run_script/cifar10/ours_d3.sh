for depth in 3
do  
    for width in "1024" "512" "256" "128" "64" "32" "16" "8" "4" "2" "1"
    do 
        echo width: $width, depth: $depth
        OMP_NUM_THREADS=1 python main.py \
            --save_root jobs/cifar10_ours \
            --dataset cifar10 \
            --max_num_path $width \
            --max_index_path $depth \
            --model_type "toyv1"
    done
done
