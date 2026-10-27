"""FairRouter: the verified Original Router-v10 + Joint/PURF V1 baseline.

The original implementation remains in enhancerouter/eargtc. This package names
and exposes those same functions; it does not introduce another algorithm copy.
"""
from enhancerouter.router import engine,raw,predict,predict_bundle,top_mask
from enhancerouter.adapter import prepare_arrays
__all__=['engine','raw','predict','predict_bundle','top_mask','prepare_arrays']
