import os
from ignite.engine import Engine
from ignite.metrics import SSIM, PSNR
from torchvision import transforms
from PIL import Image
import argparse

def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) 

def image_pair_data_loader(reference_folder, reconstructed_folder):
    ref_images = sorted(os.listdir(reference_folder))
    recon_images = sorted(os.listdir(reconstructed_folder))

    for ref, recon in zip(ref_images, recon_images):
        ref_path = os.path.join(reference_folder, ref)
        recon_path = os.path.join(reconstructed_folder, recon)

        y_true = load_image(ref_path)
        y_pred = load_image(recon_path)

        yield y_true, y_pred

def evaluate_step(engine, batch):
    y_true, y_pred = batch
    return y_pred, y_true


def main(reference_folder, reconstructed_folder):
    evaluator = Engine(evaluate_step)
    print('evaluator set')

    ssim = SSIM(data_range=1.0)  # Assuming images are normalized to [0, 1]
    psnr = PSNR(data_range=1.0)
    print('SSIM and PSNR set')

    ssim.attach(evaluator, "SSIM")
    psnr.attach(evaluator, "PSNR")

    data_loader = image_pair_data_loader(reference_folder, reconstructed_folder)
    evaluator.run(data_loader, max_epochs=1)

    metrics = evaluator.state.metrics
    print(f"SSIM: {metrics['SSIM']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SSIM and PSNR between two folders of images.")
    parser.add_argument("--reference_folder", type=str, required=True, help="Path to the folder containing reference images.")
    parser.add_argument("--reconstructed_folder", type=str, required=True, help="Path to the folder containing reconstructed images.")

    args = parser.parse_args()
    main(args.reference_folder, args.reconstructed_folder)