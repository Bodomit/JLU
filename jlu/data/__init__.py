from .utkface import UTKFace
from .vggface2 import VGGFace2

def load_datamodule(datamodule: str, **kwargs) -> UTKFace:
    if datamodule.lower() == "utkface":
        return UTKFace(**kwargs)
    elif datamodule.lower() == "vggface2":
        return VGGFace2(**kwargs)
    else:
        raise ValueError
