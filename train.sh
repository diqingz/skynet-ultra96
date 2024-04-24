PYTHON="/home/aperture/anaconda3/envs/torch/bin/python"

DATE=`date +%Y-%m-%d`

epochs=200
batch_size=32
img_size=320
device=0
learning_rate=0.001
name="UltraNet"
save_folder=${DATE}_${name}_${img_size}_0
weights="./weights/4w4a_8firstlast/test_best_7044.pt"

mkdir ./weights/${save_folder}

$PYTHON train.py \
        --epochs ${epochs} \
        --batch-size ${batch_size} \
        --img-size ${img_size} \
        --device ${device} \
        --name ${DATE} \
        --save-folder ${save_folder} \
        --lr ${learning_rate} \
        --multi-scale \
        --weights ${weights} 
