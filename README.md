# Model Agnostic Sample Reweighting for Out-of-Distribution Learning

## Requirements:

```
Pytorch 1.7
Python 3.7.7
CUDA Version 10.1
pyyaml 5.3.1
tensorboard 2.2.1
torchvision 0.5.0
tqdm 4.50.2
```

## Command
Below are the commands for replicating the results of IRM experiments. Sorry for the late release.

```bash
CUDA_VISIBLE_DEVICES=0 python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type irmv1

CUDA_VISIBLE_DEVICES=0  python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type rex
```
## Cite
If you find this implementation is helpful to your work, please cite 

```BibTeX
@inproceedings{zhou2022model,
  title={Model Agnostic Sample Reweighting for Out-of-Distribution Learning},
  author={Zhou, Xiao and Lin, Yong and Pi, Renjie and Zhang, Weizhong and Xu, Renzhe and Cui, Peng and Zhang, Tong},
  booktitle={International Conference on Machine Learning},
  pages={27203--27221},
  year={2022},
  organization={PMLR}
}
```


