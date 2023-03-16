python train.py  --batch_size 16 \
                --epochs 20 \
                --train_on kaggle \
                --checkpoint_dir checkpoints \
                --path_data data/train \
                --num_workers 2 \
                --name_model resnet50 \
                --load_height 224 \
                --load_width 128 \
                --lr 0.001 \
                --nb_classes 2 \
                --num_save_ckpt 5 \
                --save_ckpt True \
                --activation linear \
                --img_input img_face \
                --resize True \
                --name_model alexnet
                
                                        

