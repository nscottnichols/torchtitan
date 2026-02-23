from torchtitan.protocols.train_spec import register_train_spec
from .agpt import get_train_spec

spec = get_train_spec()
register_train_spec("ezpz.agpt", spec)
