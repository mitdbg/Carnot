from carnot.optimizer.rules import AddContextsBeforeComputeRule as _AddContextsBeforeComputeRule
from carnot.optimizer.rules import (
    BasicSubstitutionRule as _BasicSubstitutionRule,
)
from carnot.optimizer.rules import (
    ImplementationRule as _ImplementationRule,
)
from carnot.optimizer.rules import (
    Rule as _Rule,
)
from carnot.optimizer.rules import (
    TransformationRule as _TransformationRule,
)

ALL_RULES = [
    _AddContextsBeforeComputeRule,
    _BasicSubstitutionRule,
    _ImplementationRule,
    _Rule,
    _TransformationRule,
]

IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, _ImplementationRule)
    and rule not in [_ImplementationRule]
]

TRANSFORMATION_RULES = [
    rule for rule in ALL_RULES if issubclass(rule, _TransformationRule) and rule not in [_TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
