"""One stdlib-only numerical representation for every training configuration."""
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math

INTEGER_FIELDS=('width','updates','warm_updates','validate_every','chunk_size','legacy_draws','inference_batch','threads')
FLOAT_FIELDS=('dropout','lr','weight_decay','pseudo_weight','keep_weight')
BOOLEAN_FIELDS=('disagreement_all','agreement_ce','residual','gold_only_control')
STRING_FIELDS=('architecture','gold_sampling','pseudo_sampling','teacher')
CONFIG_FIELDS=frozenset(INTEGER_FIELDS+FLOAT_FIELDS+BOOLEAN_FIELDS+STRING_FIELDS+('clip_norm',))


def canonical_config(value):
    """Normalize only representation; TrainConfig retains its semantic validator.

    bool never substitutes for a number. Discrete fields must be actual ints;
    continuous fields accept Python int/float, become finite float, and map both
    signed zeros to positive 0.0. The input is never mutated.
    """
    if is_dataclass(value) and not isinstance(value,type): value=asdict(value)
    if not isinstance(value,Mapping) or set(value)!=CONFIG_FIELDS:
        raise ValueError('invalid training config schema')
    result=dict(value)
    for name in INTEGER_FIELDS:
        if type(value[name]) is not int: raise ValueError(name+' must be a strict integer')
    for name in BOOLEAN_FIELDS:
        if type(value[name]) is not bool: raise ValueError(name+' must be Boolean')
    for name in STRING_FIELDS:
        if type(value[name]) is not str: raise ValueError(name+' must be a string')
    for name in FLOAT_FIELDS+('clip_norm',):
        raw=value[name]
        if name=='clip_norm' and raw is None: continue
        if type(raw) not in (int,float): raise ValueError(name+' must be numeric, not Boolean')
        try: number=float(raw)
        except (OverflowError,ValueError): raise ValueError(name+' must be finite') from None
        if not math.isfinite(number): raise ValueError(name+' must be finite')
        result[name]=0.0 if number==0 else number
    return result


def canonical_pair(pair):
    if not isinstance(pair,Mapping) or set(pair)!={'g','l'}:
        raise ValueError('branch pair must contain exactly g and l')
    return {branch:canonical_config(pair[branch]) for branch in ('g','l')}
