python train.py  --batch_size 16 \
                --num_workers 2 \
                --epochs 20 \
                --name_model resnet50 \
                --load_height 224 \
                --load_width 128 \
                --checkpoint_dir checkpoints \
                --lr 0.001 \
                --path_data data/train \
                --nb_classes 2 \
                --num_save_ckpt 5 \
                --save_ckpt True \
                --activation linear \
                --train_on ssh \
                --img_input img_face 
                
                                        

