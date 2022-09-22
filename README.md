# MAPLE

#Commands

CUDA_VISIBLE_DEVICES=5  python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type irmv1

CUDA_VISIBLE_DEVICES=5  python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type rex