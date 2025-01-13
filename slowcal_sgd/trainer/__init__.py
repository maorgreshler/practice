from .sync_trainer import LocalSGD, SLowcalSGD

TRAINER_REGISTRY = {
    'LocalSGD': LocalSGD,
    'MinibatchSGD': LocalSGD,
    'SLowcalSGD': SLowcalSGD,
}


