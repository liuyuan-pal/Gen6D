from network.detector import Detector
from network.refiner import VolumeRefiner
from network.selector import ViewpointSelector

name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'selector': ViewpointSelector,
}