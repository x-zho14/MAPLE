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
Below are the commands for replicating the results of IRM experiments.

```bash
CUDA_VISIBLE_DEVICES=0 python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type irmv1

CUDA_VISIBLE_DEVICES=0 python cnn_mnist_probability_1step_irm.py --coreset_size 15000 --train_epoch 150 --max_outer_it 30 --outer_lr 1.5 --batch_size 50000 --limit 50000 --iterative --start_coreset_size 15000 --score_update --irm_type rex
```
## Cite
If you find this implementation is helpful to your work, please cite 

```BibTeX

@InProceedings{pmlr-v162-zhou22e,
  title = 	 {Sparse Invariant Risk Minimization},
  author =       {Zhou, Xiao and Lin, Yong and Zhang, Weizhong and Zhang, Tong},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {27222--27244},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/zhou22e/zhou22e.pdf},
  url = 	 {https://proceedings.mlr.press/v162/zhou22e.html},
  abstract = 	 {Invariant Risk Minimization (IRM) is an emerging invariant feature extracting technique to help generalization with distributional shift. However, we find that there exists a basic and intractable contradiction between the model trainability and generalization ability in IRM. On one hand, recent studies on deep learning theory indicate the importance of large-sized or even overparameterized neural networks to make the model easy to train. On the other hand, unlike empirical risk minimization that can be benefited from overparameterization, our empirical and theoretical analyses show that the generalization ability of IRM is much easier to be demolished by overfitting caused by overparameterization. In this paper, we propose a simple yet effective paradigm named Sparse Invariant Risk Minimization (SparseIRM) to address this contradiction. Our key idea is to employ a global sparsity constraint as a defense to prevent spurious features from leaking in during the whole IRM process. Compared with sparisfy-after-training prototype by prior work which can discard invariant features, the global sparsity constraint limits the budget for feature selection and enforces SparseIRM to select the invariant features. We illustrate the benefit of SparseIRM through a theoretical analysis on a simple linear case. Empirically we demonstrate the power of SparseIRM through various datasets and models and surpass state-of-the-art methods with a gap up to 29%.}
}

```


