# ====================================================================== #
# In this script we apply the "forward" measurement models as defined in 
# https://openreview.net/forum?id=OnD9zGAGT0k
# ====================================================================== #
import os
import torch
import torch.nn.functional as F
import yaml   
from .blur_models.kernel_encoding.kernel_wizard import KernelWizard
from .motionblur import Kernel
from torchvision.transforms.v2 import ElasticTransform

class NoiseProcess:
    """Noise class with additive Gaussian / Poisson noise
    Each measurement model inherits the noiser and forward_noise methods 
    
    Parameters
    ----------
        - noise_model: Gaussian or Poisson noise
        - noise_model: Gaussian or Poisson noise, for Poisson-noise we map the inputs to [0,1] range.
        - sigma: stdev for gaussian distribution
    """
    def __init__(self, noise_model="gaussian", sigma: float = 0.05):
        if noise_model not in ["gaussian", "poisson"]:
            raise ValueError(f"Noise model {noise_model} not implemented! Use 'gaussian' or 'poisson'.")
        self.noise_model = noise_model
        self.sigma = sigma

    def noiser(self, tensor, device=None):
        device = tensor.device if device is None else device
        if self.noise_model == "gaussian":
            noise = torch.randn_like(tensor, device=device) * self.sigma
            return tensor + noise
        elif self.noise_model == "poisson":
            tensor = (tensor - tensor.min())/tensor.max()
            return torch.poisson(tensor)

    def forward_noise(self, tensor):
        tensor = self(tensor)
        return self.noiser(tensor)

class Identity(NoiseProcess):
    "Implements the identity function as forward measurement model"
    def __init__(self, noise_model="gaussian", sigma=.05):
        super().__init__(noise_model, sigma)
        
    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return self.__class__.__name__

class RandomInpainting(NoiseProcess):
    """ 
    Implements the random-inpainting forward measurement model
    - y ~ N(Px, sigma**2 * I) if noise_model = "gaussian",
    - y ~ Poisson(Px) if noise_model = "poisson".
    
    Here P is the masking matrix given by randomly dropping 92% of pixels
    """
    def __init__(self, noise_model="gaussian", sigma=.05, inpainting_noise_level=.92):
        super().__init__(noise_model, sigma)
        self.mask = None
        self.noise_level = inpainting_noise_level
        
    def __call__(self, tensor):
        device = tensor.device
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        b, c, h, w = tensor.shape
        if self.mask is None:
            mask = (torch.rand((b, 1, h, w), device=device) > self.noise_level)
            self.mask = mask.expand(-1, c, -1, -1)
        tensor = tensor * self.mask.to(device)
        return tensor.squeeze(0) if (len(tensor.shape) == 4) and (tensor.shape[0] == 1) else tensor
    
    def __repr__(self):
        return self.__class__.__name__

class BoxInpainting(NoiseProcess):
    """ 
    Implements the box inpainting forward measurement model
    - y ~ N(y|Px, sigma**2 * I) if noise_model = "gaussian"
    - y ~ Poisson(Px; lamb) if noise_model = "poisson"
    """
    def __init__(self, noise_model="gaussian", sigma=1.):
        super().__init__(noise_model, sigma)
        self.x1 = None
        self.x2 = None
        self.box_h = None
        self.box_w = None
        self.box_values = False 

    def box(self, x):
        """Generate random coordinates for a 128x128 box that fits within the image"""
        # Only generate box values the first time. Needs to have consistent
        # forward operation for all steps. 
        if self.box_values: 
            return
        
        _, _, h, w = x.shape

        max_x = h - 128 if h >= 128 else 0
        max_y = w - 128 if w >= 128 else 0
        
        self.x1 = torch.randint(0, max(1, max_x), (1,)).item()
        self.x2 = torch.randint(0, max(1, max_y), (1,)).item()
        
        self.box_h = min(128, h)
        self.box_w = min(128, w)

        self.box_values = True
        
        return

    def __call__(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
            
        self.box(tensor)
        device = tensor.device
        b, c = tensor.shape[:2]
        
        mask = (torch.zeros((b, 1, 128, 128), device=device) > 0.5)
        mask = mask.expand(-1, c, -1, -1)
        
        tensor[:, :, self.x1:self.x1 + self.box_h, self.x2:self.x2 + self.box_w] *= mask
        
        return tensor.squeeze(0) if (len(tensor.shape) == 4) and (tensor.shape[0] == 1) else tensor
    
    def __repr__(self):
        return self.__class__.__name__


class SuperResolution(NoiseProcess):
    """
    Implementation of super resolution with bicubic downscaling and nearest-neighbor upsampling
    """
    def __init__(self, downscale_factor=0.25, upscale_factor=4, noise_model="gaussian", sigma=0.05):
        super().__init__(noise_model, sigma)
        self.downscale_factor = downscale_factor
        self.upscale_factor = upscale_factor

    def downsample(self, image):
        """
        Downsample the image using bicubic interpolation.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        downsampled = F.interpolate(
            image,
            scale_factor=self.downscale_factor,
            mode='bicubic',
            align_corners=False
        )
        return downsampled.squeeze(0) if image.shape[0] == 1 else downsampled

    def upsample(self, image):
        """
        Upsample the image using nearest neighbor interpolation to preserve
        the pixelated appearance of the downsampled image.
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        upsampled = F.interpolate(
            image,
            scale_factor=self.upscale_factor,
            mode="nearest" 
        )
        return upsampled.squeeze(0) if image.shape[0] == 1 else upsampled

    def __call__(self, tensor):
        """
        Apply the pixelation process to the input tensor.
        """
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.device.type == "mps":
            tensor = tensor.cpu()
            downsampled = self.downsample(tensor)
            upsampled = self.upsample(downsampled)
            upsampled = upsampled.to("mps")
        else:
            downsampled = self.downsample(tensor)
            upsampled = self.upsample(downsampled)

        return upsampled.squeeze(0) if len(upsampled.shape) == 4 and upsampled.shape[0] == 1 else upsampled
    
    def __repr__(self):
        return self.__class__.__name__

class NonLinearBlur(NoiseProcess):
    """
    Implements the non-linear blurring forward measurement model.
    - y ~ N(y| F(x,k), sigma**2 * I) if self.noise_model = "gaussian"
    - y ~ Poisson(F(x,k)) if self.noise_model = "poisson"
    """
    def __init__(self, noise_model="gaussian", sigma=.05):
        super().__init__(noise_model, sigma)
        self.kernel = None
        self.model = None
        
    def initialize_model(self, device):
        """Initialize the blur model if not already loaded"""
        if self.model is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yml_path = os.path.join(current_dir, "blur_models", "default.yml") 

            with open(yml_path, "r") as f:
                opt = yaml.load(f, Loader=yaml.SafeLoader)["KernelWizard"]
                model_path = opt["pretrained"]
            
            self.model = KernelWizard(opt)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.model = self.model.to(device)

    def generate_blur(self, tensor):
        """Apply blur using stored kernel or generate one if first call"""
        device = tensor.device
        self.initialize_model(device)
        
        if self.kernel is None:
            self.kernel = torch.randn((1, 512, 2, 2), device=device) * 1.2
        else:
            self.kernel = self.kernel.to(device)

        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        tensor = tensor.contiguous()
        if tensor.device != next(self.model.parameters()).device:
            self.model = self.model.to(tensor.device)
        LQ_tensor = self.model.adaptKernel(tensor, kernel=self.kernel)

        return LQ_tensor

    def __call__(self, tensor):
        input_was_4d = len(tensor.shape) == 4
        if input_was_4d:
            if tensor.shape[0] != 1:
                raise ValueError(f"Batch size must be 1, got shape {tensor.shape}")
            tensor = tensor.squeeze(0)
        result = self.generate_blur(tensor)
        if input_was_4d:
            result = result.unsqueeze(0)
            
        return result
    
    def __repr__(self):
        return self.__class__.__name__
    

class GaussianBlur(NoiseProcess):
    """
    Implements the Gaussian convolution (Gaussian noise) forward measurement model.
    The Gaussian kernel is 61x61 and convolved with the ground truth image to produce 
    the measurement. 
    """
    def __init__(self, noise_model='gaussian', kernel_size=(61,61), sigma_in_conv=3, sigma=.05):
        super().__init__(noise_model, sigma)
        self.kernel_size = kernel_size
        self.sigma_in_conv = sigma_in_conv

    def gaussian_kernel(self, device=None):
        size = self.kernel_size[0]
        x = torch.linspace(-(size // 2), size // 2, size, device=device)
        y = torch.linspace(-(size // 2), size // 2, size, device=device)
        x, y = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma_in_conv**2))
        return kernel / kernel.sum()

    def __call__(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        kernel = self.gaussian_kernel(tensor.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(tensor.size(1), 1, 1, 1)
        
        blurred = F.conv2d(tensor, weight=kernel, padding=self.kernel_size[0] // 2, groups=tensor.size(1))
        return blurred.squeeze(0) if len(blurred.shape) == 4 and blurred.shape[0] == 1 else blurred
    
    def __repr__(self):
        return self.__class__.__name__

class MotionBlur(NoiseProcess):
    """
    Implements the motion blur forward measurement model. 
    The motion blur kernel is an external kernel from (see link)
    - https://github.com/LeviBorodenko/motionblur/tree/master
    """
    def __init__(self, noise_model="gaussian", kernel_size=(61,61), intensity=0.5, sigma=.05):
        super().__init__(noise_model, sigma)
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.kernel_tensor = None

    def __call__(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if self.kernel_tensor is None:
            kernel_matrix = Kernel(size=self.kernel_size, intensity=self.intensity).kernelMatrix
            kernel_tensor = torch.tensor(kernel_matrix, dtype=tensor.dtype, device=tensor.device)
            kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)
            kernel_tensor = kernel_tensor.repeat(tensor.size(1), 1, 1, 1)
            self.kernel_tensor = kernel_tensor

        self.kernel_tensor = self.kernel_tensor.to(tensor.device)
        blurred = F.conv2d(tensor, weight=self.kernel_tensor, padding=self.kernel_size[0] // 2, groups=tensor.size(1))
        return blurred.squeeze(0) if len(tensor.shape) == 4 and tensor.shape[0] == 1 else blurred
    
    def __repr__(self):
        return self.__class__.__name__
    
class PhaseRetrieval(NoiseProcess):
    """
    Implements the phase retrieval forward measurement model:
    - y ~ N(y||FPx_0|, σ²I) for Gaussian noise
    - y ~ P(y||FPx_0|; λ) for Poisson noise
    
    where:
    F = 2D Discrete Fourier Transform 
    P = Oversampling matrix with ratio k/n
    |·| = magnitude of complex number

    We compute the oversampling by padding with torch.nn.functional.pad, which is equivalent to oversampling
    - https://ccrma.stanford.edu/~jos/dft/Zero_Padding_Theorem_Spectral.html
    """
    def __init__(self, noise_model="gaussian", sigma=0.05, upscale_factor=4.):
        super().__init__(noise_model, sigma)
        self.upscale_factor = upscale_factor
        self.padding = int((upscale_factor / 8.0) * 256)
        
    def apply_oversampling(self, tensor):
        """Applies oversampling matrix P with ratio k/n via padding"""
        return F.pad(tensor, (self.padding, self.padding, self.padding, self.padding))
    
    def __call__(self, tensor):
        """Computes the Magnitude of the centered Fourier coefficients"""
        tensor_pad = self.apply_oversampling(tensor)
        if tensor.device.type == "mps":
            tensor_pad = tensor_pad.to("cpu")
        fourier_coeffs = torch.fft.fft2(tensor_pad, norm="ortho")
        fourier_coeffs = torch.fft.fftshift(fourier_coeffs)
        fourier_coeffs = fourier_coeffs.to(tensor.device)
        return fourier_coeffs.abs()

    def __repr__(self):
        return self.__class__.__name__
    

class Magnitude(NoiseProcess):
    """
    Implements the magnitude forward measurement model:
    - y ~ N(y||x_0|, σ²I) for Gaussian noise
    - y ~ P(y||x_0|; λ) for Poisson noise
    
    where:
    |·| = magnitude of complex number
    """
    def __init__(self, noise_model="gaussian", sigma=0.05):
        super().__init__(noise_model, sigma)
    
    def __call__(self, tensor):
        """Computes the Magnitude of the centered Fourier coefficients"""
        return tensor.abs()

    def __repr__(self):
        return self.__class__.__name__
    

class Grayscale(NoiseProcess):
    """
    Implements grayscale conversion as a forward measurement model.
    Converts RGB images to grayscale using standard luminance weights:
    - Red: 0.2989
    - Green: 0.5870
    - Blue: 0.1140
    """
    def __init__(self, noise_model="gaussian", sigma=0.05):
        super().__init__(noise_model, sigma)
        # Standard weights for RGB to grayscale conversion
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140])
        
    def __call__(self, tensor):
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        self.weights = self.weights.to(tensor.device)
        if tensor.shape[1] == 3: 
            weights = self.weights.view(1, 3, 1, 1)
            grayscale = (tensor * weights).sum(dim=1, keepdim=True)
            grayscale = grayscale.repeat(1, 3, 1, 1)
        else:  
            return tensor
            
        return grayscale.squeeze(0) if (len(grayscale.shape) == 4) and (grayscale.shape[0] == 1) else grayscale
    
    def __repr__(self):
        return self.__class__.__name__
    

class RandomElastic(NoiseProcess):
    """
    Implements the Randomly Elastic forward measurement model (Not differentiable - so this does not work yet)
    """
    def __init__(self, noise_model="gaussian", sigma = 0.05):
        super().__init__(noise_model, sigma)
        self.elastic = None
        self.alpha = 40
        self.elastic_sigma = 10

    def __call__(self, tensor):
        tensor_ = tensor.clone()
        if len(tensor_.shape) == 3:
            tensor_ = tensor_.unsqueeze(0)
        
        if self.elastic is None:
            self.elastic = ElasticTransform(self.alpha, self.elastic_sigma)

        tensor_ = self.elastic(tensor_)
        return tensor_.squeeze(0) if (len(tensor_.shape) == 4) and (tensor_.shape[0] == 1) else tensor_
    
    def __repr__(self):
        return self.__class__.__name__
