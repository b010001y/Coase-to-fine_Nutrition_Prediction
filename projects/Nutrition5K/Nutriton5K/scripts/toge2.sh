#!/bin/bash
for segment in 20 25 ;do
{
  #python main.py --CUDA_DEVICE 1 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment 10 --epoch 30 --alpha 1 --beta 1 --threshold 0.1 --seed 0 &
  python main.py --CUDA_DEVICE 0 --batch_size 64 --output_dim 4096 --found_lr 1e-4 --segment $segment --epoch 30 --alpha 1 --beta 1 --threshold 0.2 --seed 5 &
wait
}
done
