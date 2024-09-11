from microsim.allen import Specimen
from microsim.util import ndview
from tifffile import imwrite
import numpy as np

spec = Specimen.fetch(555241040)
swc = spec.neuron_reconstructions[0].getSomaLocation()
print(swc)