"""Update rules of the four models: memberships M (update_mu), weights h (update_w) and centres P (update_v).

    method       class                equations
    SPMVKFCM     MVKFC                (5b)-(8)
    SPMVFCM      MVJFC                (1)-(4)
    SPWPMVFCM    WEIGHT_POINT_MVJFC   (9)-(12)
    SPWPMVKFCM   WEIGHT_POINT_MVKFC   (13b)-(16)

The iterations (P -> h -> M), the objective and the stopping test are in spmvfcm/methods.py.
Data layout: x (V, H, W) views; v (V, C) centres; u (C, H, W) memberships; w (V,) view weights (SPMVFCM, SPMVKFCM)
or (V, H, W) point-wise weights (SPWPMVFCM, SPWPMVKFCM).
"""
import numpy as np

# ---- SPMVKFCM: Eqs. (5b)-(8) ------------------------------------------------------------
class MVKFC:
    def __init__(self, x, w, v, u, C, param=1e+11, div_error_corr=1e-10, iterations=10, fuzzy=2, sigma=0.5):
        self.x = x
        self.w = w
        self.v = v
        self.u = u
        self.C= C
        self.fuzzy=fuzzy
        self.sigma=sigma
        self.param = param
        self.div_error_corr = div_error_corr
        self.loss_margin=[]
        self.iterations = iterations

    def update_w(self):
        #completed
        diff=1-np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2))
        #weighted_w=np.sum(tuple( 2 * diff[i] * (self.u[i]** self.fuzzy) for i in range(self.C) ), axis=0 )
        #numenator=np.exp( (-np.sum (weighted_w,axis=(1,2))) /self.param )
        weighted_w=np.sum(tuple( 2 * diff[i] * (self.u[i]** self.fuzzy) for i in range(self.C) ), axis=(0,2,3) )
        numenator=np.exp( (-1 * weighted_w) /self.param )
        return numenator/np.sum(numenator)


    def update_mu(self):
        #completed
        diff=1-np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2))
        numenator=np.sum(diff * self.w[np.newaxis,:,np.newaxis,np.newaxis],axis=1)
        denominator=np.sum(numenator, axis=0)
        updated_u = np.power(1/(numenator/denominator), 1/(self.fuzzy-1))
        norm=np.sum(updated_u,axis=0)
        return updated_u/norm


    def update_v(self):
        #completed
        diff=np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2))
        numenator=np.sum(diff * self.x * ( self.u[:,np.newaxis] ** self.fuzzy), axis=(2,3))
        denominator=np.sum(diff * (self.u[:,np.newaxis]**self.fuzzy),axis=(2,3))
        return (numenator/denominator).T


# ---- SPMVFCM: Eqs. (1)-(4) ------------------------------------------------------------


class MVJFC:
    def __init__(self, x, w, v, u, C, param=1e+11, div_error_corr=0.000001, iterations=10, fuzzy=2):
        self.x = x
        self.w = w
        self.v = v
        self.u = u
        self.C= C
        self.fuzzy=fuzzy
        self.param = param
        self.div_error_corr = div_error_corr
        self.loss_margin=[]
        self.iterations = iterations

    def update_w(self):
        diff=np.stack( tuple( (self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2  for i in range(self.C)) )
        weighted_w=np.sum(  tuple( diff[i] * (self.u[i]** self.fuzzy) for i in range(self.C) ), axis=0 )
        numenator=np.exp( (-np.sum (weighted_w,axis=(1,2))) /self.param )
        return numenator/np.sum(numenator)


    def update_mu(self):
        diff=np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis]) **2  for i in range(self.C)) )

        numenator=np.sum(diff * self.w[np.newaxis,:,np.newaxis,np.newaxis],axis=1)
        denominator=np.sum(numenator, axis=0)
        updated_u = np.power(1/(numenator/denominator), 1/(self.fuzzy-1))
        norm=np.sum(updated_u,axis=0)
        return updated_u/norm


    def update_v(self):
        return np.stack(tuple( (np.sum(self.x * (self.u[i] ** self.fuzzy), axis=(1,2))/np.sum(self.u[i] ** (self.fuzzy))) for i in range(self.C))).T


# ---- SPWPMVFCM: Eqs. (9)-(12) ------------------------------------------------------------
class WEIGHT_POINT_MVJFC:
                   
    def __init__(self, x, w, v, u, C, param=1e+11, div_error_corr=1e-10, iterations=10, fuzzy=2, sigma=0.5):
        self.x = x
        self.w = w
        self.v = v
        self.u = u
        self.C= C
        self.fuzzy=fuzzy
        self.sigma=sigma
        self.param = param
        self.div_error_corr = div_error_corr
        self.loss_margin=[]
        self.iterations = iterations

         #code updated.......
    def update_mu(self):
        diff=np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis]) **2  for i in range(self.C)) )
        numenator=np.sum(diff * self.w,axis=1)
        denominator=np.sum(numenator, axis=0)
        updated_u = np.power(1/(numenator/denominator), 1/(self.fuzzy-1))
        norm=np.sum(updated_u,axis=0)
        return updated_u/norm
    
    
    #code updated.......
    def update_w(self):
        #completed
        diff=np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis]) **2  for i in range(self.C)) )
        weighted_w=diff*np.expand_dims(self.u**self.fuzzy, axis=1)
        numenator=np.exp(-1*(np.sum(weighted_w,axis=0)/self.param))
        return numenator/np.sum(numenator,axis=0)
    
    
    
    
    #code updated.......
    def update_v(self):
        denominator=np.expand_dims(self.u**self.fuzzy,axis=1)*self.w
        numenator=np.sum(self.x * denominator,axis=(2,3))
        denominator=np.sum(denominator,axis=(2,3))
        return (numenator/denominator).T


    


# ---- SPWPMVKFCM: Eqs. (13b)-(16) ------------------------------------------------------------
class WEIGHT_POINT_MVKFC:
                   
    def __init__(self, x, w, v, u, C, param=1e+11, div_error_corr=1e-10, iterations=10, fuzzy=2, sigma=0.5):
        self.x = x
        self.w = w
        self.v = v
        self.u = u
        self.C= C
        self.fuzzy=fuzzy
        self.sigma=sigma
        self.param = param
        self.div_error_corr = div_error_corr
        self.loss_margin=[]
        self.iterations = iterations

         #code updated.......
    def update_mu(self):
        diff=1-np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2))
        numenator=np.sum(diff * self.w,axis=1)
        denominator=np.sum(numenator, axis=0)
        updated_u = np.power(1/(numenator/denominator), 1/(self.fuzzy-1))
        norm=np.sum(updated_u,axis=0)
        return updated_u/norm
    
    
    
    #code updated.......
    #code updated.......
    def update_w(self):
    #completed
        diff=2*(1-np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2)))
        weighted_w=diff*np.expand_dims(self.u**self.fuzzy, axis=1)
        numenator=np.exp(-1*(np.sum(weighted_w,axis=0)/self.param))
        return numenator/np.sum(numenator,axis=0)
        
    
    
    #code updated.......
    #code updated.......
    def update_v(self):
        ker=np.exp(-np.stack( tuple((self.x - self.v[:,i][:,np.newaxis,np.newaxis])**2   for i in range(self.C)) )/(self.sigma**2))
        denominator=np.expand_dims(self.u**self.fuzzy,axis=1)*self.w*ker
        numenator=np.sum(self.x * denominator,axis=(2,3))
        denominator=np.sum(denominator,axis=(2,3))
        return (numenator/denominator).T


    
