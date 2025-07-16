import os
import random

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFont, ImageDraw

from helpers.utils import add_watermark



# from WMEmbeddedSystems
class RandomWatermark(object):
	# noinspection PyUnresolvedReferences
	"""Normalize an tensor image with mean and standard deviation.
		Given mean: ``(M1,...,Mn)`` and std: ``(M1,..,Mn)`` for ``n`` channels, this transform
		will normalize each channel of the input ``torch.*Tensor`` i.e.
		``input[channel] = (input[channel] - mean[channel]) / std[channel]``
		Args:
			mean (sequence): Sequence of means for each channel.
			std (sequence): Sequence of standard deviations for each channel.
		"""

	def __init__(self, watermark, probability=0.5):
		self.watermark = torch.from_numpy(watermark)
		self.probability = probability

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized Tensor image.
		"""
		if random.random() < self.probability:
			return add_watermark(tensor, self.watermark)
		return tensor

class EmbedText(object):

	def __init__(self, text, pos, strength):
		self.text = text
		self.pos = pos
		self.strength = strength

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized Tensor image.
		"""
		image = transforms.ToPILImage()(tensor)

		draw = ImageDraw.Draw(image)

		font_path = os.path.join(os.getcwd(), "font", "sans_serif.ttf")
		font = ImageFont.truetype(font_path, 10)

		draw.text(self.pos, self.text, fill=int(255*self.strength), font=font)
		#image.show()
		tensor = transforms.ToTensor()(image)

		return tensor

class EmbedPattern(object):
    """
    Embeds a predefined pattern into an image tensor.
    The pattern consists of white squares placed in the four corners of the image.
    """
    def __init__(self, mode='grayscale'):
        """
        Initializes the pattern based on the specified mode.

        Args:
            mode (str): 'grayscale' for single-channel images (e.g., MNIST) or 
                        'rgb' for three-channel images (e.g., CIFAR-10).
        """
        self.mode = mode
        
        # Determine image and pattern parameters based on the mode
        if mode == 'grayscale':
            image_size = 28  # MNIST image size
            num_channels = 1
            pattern_size = 2 # Size of the corner squares
        else:  # Assumes 'rgb'
            image_size = 32  # CIFAR-10 image size
            num_channels = 3
            pattern_size = 3

        # Create an empty tensor for the pattern
        self.pattern = torch.zeros((num_channels, image_size, image_size))

        # Add white squares (value 1.0) to the four corners of the pattern
        # Top-left corner
        self.pattern[:, 0:pattern_size, 0:pattern_size] = 1.0
        # Top-right corner
        self.pattern[:, 0:pattern_size, -pattern_size:] = 1.0
        # Bottom-left corner
        self.pattern[:, -pattern_size:, 0:pattern_size] = 1.0
        # Bottom-right corner
        self.pattern[:, -pattern_size:, -pattern_size:] = 1.0

    def __call__(self, tensor):
        """
        Applies the watermark pattern to the input image tensor.

        Args:
            tensor (Tensor): An image tensor of size (C, H, W).
        Returns:
            Tensor: The watermarked image tensor.
        """
        # Ensure the pattern is on the same device as the input tensor
        if self.pattern.device != tensor.device:
            self.pattern = self.pattern.to(tensor.device)
            
        # Add the pattern to the tensor and clamp the values to the valid [0, 1] range
        watermarked_tensor = torch.clamp(tensor + self.pattern, 0, 1)
        
        return watermarked_tensor
