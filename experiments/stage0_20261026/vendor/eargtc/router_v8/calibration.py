"""Low capacity, identity-shrunk calibration without policy/test feedback."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_affine(logits, target, *, bias_only=False, strength=2., min_per_class=5):
    x, y = np.asarray(logits,dtype=float),np.asarray(target,dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('calibration inputs must be finite aligned vectors')
    counts = [int((y==k).sum()) for k in (0,1)]
    if len(y)==0:
        return dict(slope=1.,bias=0.,mode='identity_empty',counts=counts,negative_slope=False)
    bias_only = bias_only or min(counts)<min_per_class
    def objective(theta):
        a,b = (1.,theta[0]) if bias_only else theta
        eta = a*x+b
        reg = .5*strength*((a-1)**2+b*b)
        value = np.logaddexp(0,eta).sum()-(y*eta).sum()+reg
        delta = expit(eta)-y
        grad = [delta.sum()+strength*b] if bias_only else [(delta*x).sum()+strength*(a-1),delta.sum()+strength*b]
        return value,np.asarray(grad)
    start = np.array([0.]) if bias_only else np.array([1.,0.])
    fit = minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'calibration failed: {fit.message}')
    a,b = (1.,float(fit.x[0])) if bias_only else map(float,fit.x)
    return dict(slope=a,bias=b,mode='bias_only' if bias_only else 'affine',counts=counts,negative_slope=a<0,strength=strength)


def apply_affine(logits, fit):
    return fit['slope']*np.asarray(logits)+fit['bias']


def fit_heads(c,r,state,**kwargs):
    state=np.asarray(state)
    decisive=state!=2
    return dict(trust=fit_affine(c,decisive.astype(float),**kwargs),
                preference=fit_affine(np.asarray(r)[decisive],(state[decisive]==0).astype(float),**kwargs))


def calibrated_logits(c,r,fits):
    return apply_affine(c,fits['trust']),apply_affine(r,fits['preference'])


def probabilities(c,r):
    c,r=expit(c),expit(r)
    return np.column_stack((c*r,c*(1-r),1-c))
