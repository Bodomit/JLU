from jlu.data.fairface import FairFace

from .celeba import CelebA
from .common import AttributeDataset
from .fairface import FairFace
from .utkface import UTKFace
from .vggface2 import VGGFace2
from .vggface2_maadface import VGGFace2WithMaadFace


def load_datamodule(datamodule: str, **kwargs):
    if datamodule.lower() == "utkface":
        return UTKFace(**kwargs)
    elif datamodule.lower() == "vggface2":
        return VGGFace2(**kwargs)
    elif datamodule.lower() == "vggface2_maadface":
        return VGGFace2WithMaadFace(**kwargs)
    elif datamodule.lower() == "fairface":
        return FairFace(**kwargs)
    elif datamodule.lower() == "celeba":
        return CelebA(**kwargs)
    else:
        raise ValueError
