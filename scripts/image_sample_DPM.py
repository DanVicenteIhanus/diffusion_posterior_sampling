"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
Source: openAI

Modified by Alexander Gutell, Dan Vicente and Ludvig Skare in 2024
"""

import os
import time
from datetime import datetime
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision import (datasets, 
                         transforms)
from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    get_measurement_model
)
import matplotlib.pyplot as plt
from diffusion.data_utils import SingleImageDataset

from dpm_solver.sampler import NoiseScheduleVP, model_wrapper
from diffusion.respace import DPMDiffusionPosteriorSampling

def normalize_np(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    return img

def process_image(x):
    """Process image for saving - handles both tensor and numpy inputs"""
    if isinstance(x, th.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    if x.ndim == 4: 
        x = x[0]
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return normalize_np(x)

def main():
    args = create_argparser()
    rank = 0 # or rank = dist.get_rank()

    # dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    dev = dist_util.dev()
    print(f"Using device = '{dev}'")

    print(f"Using measurement model: {args.measurement_model}")

    model.to(dev)
    model.eval()

    t_start = datetime.now()

    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        if args.dps_update:
            # ============================================ #
            # Perform the DPS sampling as in algorithm 1/2 #
            # Additional DPM-acceleration in sampling      #
            # ============================================ #
            measurement_model = get_measurement_model(
                name=args.measurement_model,
                noise_model=args.noise_model,
                sigma=args.sigma,
                inpainting_noise_level=args.inpainting_noise_level
            )

            #dataset = args.data_path.split("/")[2]
            output_dir = args.output_dir
            #output_dir = f"./output/{measurement_model}/{dataset}"
            logger.configure(dir=output_dir)
            #logger.configure(dir=f"./output/{measurement_model}/{dataset}/{time.time()}")
            logger.log(f"Preparing dataset with {measurement_model} and creating dps sampler with {args.timestep_respacing} respaced steps...")
            
            # Compute the forward measurement + noise
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            if args.single_image_data:
                dataset = SingleImageDataset(args.data_path, transform=transform)
            else:
                dataset = datasets.ImageFolder(args.data_path, transform=transform)

            dataloader = th.utils.data.DataLoader(
                dataset,
                batch_size=args.sampling_batch_size, 
                shuffle=False
            )

            imgs_clean, _ = next(iter(dataloader))
            img_clean = imgs_clean[-1].requires_grad_(False)
            img_noisy = measurement_model.forward_noise(img_clean.clone())

            # Save clean + noisy images for reference
            for prefix, img in [("clean", img_clean), ("meas", img_noisy)]:
                #img = img.clone() # Note that processing is done in place, so if we clone here we do not get the min-max normalization
                img_processed = process_image(img)
                img_dir = os.path.join(output_dir, prefix)
                os.makedirs(img_dir, exist_ok=True)
                save_path = os.path.join(img_dir, args.img_name)
                plt.imsave(save_path, img_processed)
                del img_processed

            #### DPM - Fast ODE sampling
            noise_schedule = NoiseScheduleVP(
                schedule="discrete", 
                betas=th.tensor(
                    diffusion.betas,
                    device=dev))
            
            # disable grads fro model parameters
            #for param in model.parameters():
            #   param.requires_grad_(False)

            def model_fn(x, t, **kwargs):
                with th.enable_grad():
                    if isinstance(t, float):
                        t = th.tensor([t], device=x.device)
                    elif len(t.shape) == 0:
                        t = t.view(-1)
                    if len(x.shape) == 3:
                        x = x.view(1, *x.shape)
                    output = model(x, t, **kwargs)
                    if output.shape[1] == 6:
                        output = output[:, :3]
                    
                    return output.requires_grad_(True)


            wrapped_model = model_wrapper(
                model=model_fn,
                noise_schedule=noise_schedule,
                model_type="noise",
                model_kwargs={},
                guidance_type="uncond"
            )

            # Create the DPS-DPM model with all necessary parameters
            dpm_diffusion = DPMDiffusionPosteriorSampling(
                model_fn=wrapped_model,
                noise_schedule=noise_schedule,
                measurement_model=measurement_model,
                measurement=img_noisy.to(dev),
                noise_model=args.noise_model,
                step_size=args.step_size,
                algorithm_type="dpmsolver++",
                gaussian_diffusion=diffusion,
                ddpm_model=model
            )

            logger.log(f"Created DPM-DPS solver with step_size = {args.step_size}..\n")
            print("Starting sampling..\n")
            sample = dpm_diffusion.sample(
                x=th.randn(1, 3, 256, 256, device=dev).requires_grad_(True),
                steps=int(args.timestep_respacing),
                order=3,
                skip_type="time_uniform",
                method="multistep"
            )
        else:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size)
            )
        
        if rank == 0:
            try:
                sample_processed = process_image(sample)
                sample_dir = os.path.join(output_dir, "sample")
                os.makedirs(sample_dir, exist_ok=True)
                save_path = os.path.join(sample_dir, args.img_name)
                plt.imsave(
                    save_path,
                    sample_processed
                )
                logger.log(f"Saved sample image to {save_path}")
            except Exception as e:
                logger.log(f"Failed to save sample image: {e}")

        sample_uint8 = (process_image(sample) * 255).astype(np.uint8)        
        if rank > 1:
            gathered_samples = [th.zeros_like(sample_uint8) for _ in range(dist.get_world_size())]
            dist_util.safe_all_gather(gathered_samples, sample_uint8)
            all_images.extend([s for s in gathered_samples])
        else:
            all_images.append(sample_uint8)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    # if rank == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")
    t_end = datetime.now()
    t_execution = (t_end - t_start)
    print(f"Elapsed time during sampling = {t_execution}")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="./models/256x256_diffusion_uncond.pt",
        dps_update=True,
        measurement_model="BoxInpainting",
        noise_model="gaussian",
        sigma=.05,
        inpainting_noise_level=0.92,
        step_size=1.,
        data_path="./datasets/eval_imgs_imagenet", # Default = ImageNet validation
        sampling_batch_size=10,
        single_image_data=True,
        img_name = "img.jpg"
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    if args.output_dir is None:
        dataset = args.data_path.split("/")[2]
        args.output_dir = f"./output/{args.measurement_model}/{dataset}/{time.time()}"
    return args

if __name__ == "__main__":
    main() 