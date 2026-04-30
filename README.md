# Joint-Reconstruction

This repository contains the MATLAB implementation for **multimodal X-ray ptychography and fluorescence reconstruction**.  
All experiments are carried out in MATLAB and optimized using the **Truncated Newton (TN)** method.

## Reproducing the Experimental Results

To reproduce the results reported in the paper, please follow the steps below.

### 1. Data Generation

Generate the simulated datasets using the following scripts:

- `data_generation.m`  
  Generates the **phantom simulation dataset**.

- `data_generation_cameraman.m`  
  Generates the **Cameraman/Baboon dataset**.

### 2. Image Reconstruction Experiments

Run the reconstruction scripts below to reproduce the main experimental results:

- `recon_image_compare.m`  
  Reproduces the reconstruction results for the **phantom simulation dataset**.

- `recon_image_compare_cameraman.m`  
  Reproduces the reconstruction results for the **Cameraman/Baboon dataset**.

  - `multiple_ne_comparison.m`  
  Reproduces the reconstruction results for the **multiple element maps dataset**.

### 3. Gradient and Hessian Analysis

Additional scripts are provided for verification and analysis of the optimization model:

- `gradient_test_true_point.m`  
  Evaluates the gradient at the **ground-truth point**.

- `loss_exact_hessian_formation.m`  
  Computes the **exact Hessian** and analyzes the **loss surface**.

## Notes

Please make sure all required data files and dependencies are properly configured before running the scripts.  
For best reproducibility, we recommend running the experiments in the same MATLAB environment and with the same optimizer settings used in the paper.



Joint Reconstruction code is developed by:

- Chengru Zou (<eric.zou@emory.edu>)
- Yuanzhe Xi (<yuanzhe.xi@emory.edu>)
- Zichao Wendy Di(<wendydi@mcs.anl.gov>)

We welcome your questions. Please contact **Chengru Zou** specifically for questions related to package usage, features, and development.

## Citation

If you use Joint reconstruction code, please cite the following paper:

```bibtex
@misc{zou2025jointvariationalframeworkmultimodal,
      title={A Joint Variational Framework for Multimodal X-ray Ptychography and Fluorescence Reconstruction}, 
      author={Eric Zou and Elle Buser and Zichao Wendy Di and Yuanzhe Xi},
      year={2025},
      eprint={2511.02153},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2511.02153}, 
}
