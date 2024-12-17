# [Re] Diffusion Posterior Sampling

This is a reimplementation of the Diffusion Posterior Sampling algorithm and a partial reproduction (and extension) of the experiments outlined in the paper by Chung et. al [1]. The implementations are done by Dan Vicente Ihanus (`dan.vicente.ihanus@gmail.com`), Alexander Gutell (`agutell@kth.se`) and Ludvig Skare (`lskare@kth.se`) as a course project in DD2412 at KTH, Royal Institute of Technology the fall of 2024. The implementation uses PyTorch and can run with GPU acceleration using `MPS` and `CUDA`.

We have tried to reimplement everything from scratch and try to offer an alternative to the official repository of the authors. The codebase is mainly based on the existing code from (Dhariwal et. Nichol [2]) https://github.com/openai/guided-diffusion. 

We also try to extend the DPS sampling procedure to the fast DPM-sampler by Lu et. al[3]. The official codebase of the DPM-samplers are found in https://github.com/LuChengTHU/dpm-solver, we mainly experiment with DPM-Solver++ of order 3.


## Getting started
To create a venv and add poetry, follow these steps (Assuming UNIX environment)
We have tested and run this code on Python>=3.11 and provide a requirements.txt file for easy pip-install.


## Clone git repo
```
git clone 
cd diffusion_posterior_sampling
```

## Create venv and install Poetry

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip setuptools
.venv/bin/pip install poetry
```

## Install dependencies:

```bash
poetry install 
```

## Alternatively
```bash
pip install requirements.txt
```

## Measurement models
Measurement models are found in `./data/measurement_models.py`. To add a new one, simply add a subclass of the `NoiseProcess` class and implement your (differentiable) `forward` method. Note that we have implemented everything in PyTorch.


## Downloading pre-trained models
The model for the `Imagenet` dataset is the 256x256 diffusion `256x256_diffusion_uncond.pt` found here:
https://github.com/openai/guided-diffusion

The model for the `FFHQ` and `celebA-HQ`datasets is the `ffhq_baseline.pt`-model found here:
https://github.com/jychoi118/ilvr_adm

Download these `.pt` files into the `./models` directory.

The model for the `NonLinearBlur` model is the pretrained `GOPRO_VAE.pth` model found here: 
https://github.com/VinAIResearch/blur-kernel-space-exploring

Download this model into the `.measurement_models/blur_models/pretrained/` folder.

## Solving Inverse Problems
Main sampling procedure is found in `./scripts/image_sample.py`. Example run with Gaussian blur, $\sigma =0.05$ on the `celebA-HQ` dataset with $NFE=1000$ and step_size = 0.5:

```bash
poetry run python scripts/image_sample.py \
            --attention_resolutions "16" \
            --class_cond "False" \
            --diffusion_steps "1000" \
            --dropout "0.0" \
            --image_size "256" \
            --learn_sigma "True" \
            --noise_schedule "linear" \
            --num_channels "128" \
            --num_head_channels "64" \
            --num_res_blocks "1" \
            --resblock_updown "True" \
            --use_fp16 "False" \
            --use_scale_shift_norm "True" \
            --model_path "./models/ffhq_baseline.pt" \
            --num_samples "1" \
            --batch_size "1" \
            --timestep_respacing "1000" \
            --dps_update "True" \
            --measurement_model "GaussianBlur" \
            --noise_model "gaussian" \
            --sigma "0.05" \
            --step_size "0.5" \
            --data_path "./datasets/eval_imgs_celeba_hq" \
            --sampling_batch_size "1" \
            --single_image_data "True" 
```

## Re-run experiments from our paper
We have wrapped the experiments from our paper in `shell`-scripts. To run all of the inverse problem experiments on the full data in `./datasets/*` run the following

- $\texttt{FFHQ}$ and $\texttt{celebA-HQ}$

```bash
./scripts/run_experiments_imagenet.sh <MeasurementModel>
```

- $\texttt{Imagenet}$ 
```bash
./scripts/run_experiments_imagenet.sh <MeasurementModel>
```

## Using the fast DPM-Sampler
Our experiments with the DPM-Sampler are in the cradle and are rather unstable. To run some sampling with $NFE=5$ (extremely quick on our systems), you can run

```bash
./scripts/sample_DPM.sh eval_imgs_imagenet imagenet RandomInpainting 115. 0.92
```

------
[1]https://openreview.net/forum?id=OnD9zGAGT0k

[2]https://arxiv.org/abs/2105.05233

[3]https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf