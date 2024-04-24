PYTHON="/home/aperture/anaconda3/envs/torch/bin/python"

DATE=`date +%Y-%m-%d`

weights="./weights/4w4a_8firstlast/test_best_7044.pt"

batch_size=32
img_size=320
device=1

$PYTHON test.py --weights ${weights} \
        --batch-size ${batch_size} \
        --img-size ${img_size} \
        --device ${device} \
