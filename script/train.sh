python train.py  --batch_size 32 \
                --num_workers 2 \
                --epochs 20 \
                --start_epoch 1 \
                --name_model alexnet \
                --load_height 224 \
                --load_width 128 \
                --checkpoint_dir checkpoints \
                --lr 0.01 \
                --path_data data/train/0 \
                --num_classes 2 \
                --save_ckpt_num 5 \
                --save_ckpt True \
                --activation sigmoid
                
                                        

