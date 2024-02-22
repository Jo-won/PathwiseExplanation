for depth in 10
do  
    for width in "32" "64" "128" "256" "512" "1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072" "262144" "524288"
    do 
        echo width: $width, depth: $depth
        OMP_NUM_THREADS=1 python main.py \
            --save_root jobs/imgNS_ours \
            --dataset imagenet_select \
            --max_num_path $width \
            --max_index_path $depth
    done
done

