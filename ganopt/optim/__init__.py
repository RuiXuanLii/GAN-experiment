from .adam import Adam
from .extraadam import ExtraAdam
from .optimistic_adam import OptimisticAdam
from .lookahead_minmax import LookaheadMinmax

METHODS = {"adam": Adam, "extraadam": ExtraAdam,
           "optimistic_adam": OptimisticAdam, "lookahead_minmax": LookaheadMinmax}
