from jlu.data.fairface import FairFace

from .fairface import FairFace
from .utkface import UTKFace
from .vggface2 import VGGFace2
<<<<<<< HEAD
from .vggface2_maadface import VGGFace2WithMaadFace
=======
from .fairface import FairFace
>>>>>>> Switch to Adam.


def load_datamodule(datamodule: str, **kwargs):
    if datamodule.lower() == "utkface":
        return UTKFace(**kwargs)
    elif datamodule.lower() == "vggface2":
        return VGGFace2(**kwargs)
<<<<<<< HEAD
    elif datamodule.lower() == "vggface2_maadface":
        return VGGFace2WithMaadFace(**kwargs)
=======
>>>>>>> Switch to Adam.
    elif datamodule.lower() == "fairface":
        return FairFace(**kwargs)
    else:
        raise ValueError
