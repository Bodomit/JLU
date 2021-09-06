from typing import Union

from .fairface import FairFace
from .utkface import UTKFace
from .vggface2 import VGGFace2


def load_datamodule(datamodule: str, **kwargs) -> Union[UTKFace, VGGFace2, FairFace]:
    if datamodule.lower() == "utkface":
        return UTKFace(**kwargs)
    elif datamodule.lower() == "vggface2":
        return VGGFace2(**kwargs)
    elif datamodule.lower() == "fairface":
        return FairFace(**kwargs)
    else:
        raise ValueError
