"""The shared temporal denoiser used by SceneMI.

The implementation remains in the established diffusion module.  This public
name makes the architecture discoverable without duplicating its parameters or
changing any checkpoint keys.
"""
from model.mdm_scene_unet import TemporalUnet, TemporalUnetLarge, MDM_Scene_UNET

__all__ = ['TemporalUnet', 'TemporalUnetLarge', 'MDM_Scene_UNET']
