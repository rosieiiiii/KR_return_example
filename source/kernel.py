import numpy as np
from sklearn.utils.extmath import randomized_svd

def generate_kernel_matrix(alpha,
    delta,
    Nmax=3650,
    Nmax_y=None):
    '''
    - Generate a kernel matrix with parameter alpha. No rows or columns correspond to infinite maturity.
    - Args:
        - alpha (float): kernel hyper-parameter alpha
        - Nmax (int): number of rows in the output kernel matrix
        - Nmax_y (int): number of columns in the output kernel matrix. If it is None, set Nmax_y=Nmax.
    - Returns:
        - K (numpy array of dim (Nmax,Nmax_y)): kernel matrix with hyper-paramter alpha
    '''
    assert 0<=delta<=1

    if 0<delta<1:
        sqrt_D=np.sqrt(alpha**2 + 4*delta/(1-delta))
        l_1=(alpha-sqrt_D)/2
        l_2=(alpha+sqrt_D)/2
    else:
        sqrt_D=l_1=l_2=None

    if Nmax_y is None:
        Nmax_y=Nmax
    K=np.full((Nmax,Nmax_y),np.nan)
    for i in range(Nmax):
        x=(i+1)/365
        arr_y=np.arange(1,Nmax_y+1)/365
            
        min_xy=np.minimum(x,arr_y)
        max_xy=np.maximum(x,arr_y)
        
        if delta==0:
            K[i,:]=\
                -min_xy/alpha**2*np.exp(-alpha*min_xy)+\
                2/alpha**3*(1-np.exp(-alpha*min_xy))-\
                min_xy/alpha**2*np.exp(-alpha*max_xy)
        elif delta==1:
            K[i,:]=\
                1/alpha*(1-np.exp(-alpha*min_xy))
        else:
            K[i,:]=\
                -alpha/(delta*l_2**2)*(1-np.exp(-l_2*x)-np.exp(-l_2*arr_y))+\
                1/(alpha*delta)*(1-np.exp(-alpha*min_xy))+\
                1/(delta*sqrt_D)*(l_1**2/l_2**2 * np.exp(-l_2*(x+arr_y)) - np.exp(-l_1*min_xy - l_2* max_xy) )

    return K

def generate_kernel_vec_inf(alpha,
    delta,
    Nmax=3650):
    '''
    - Generate Nmax-dimensional vector k(x,infty) for x=1/365,....,Nmax/365
    - Args:
        - alpha (float): kernel hyper-parameter alpha
        - Nmax (int): dimension of the output vector
    - Returns:
        - arr_k_inf (numpy array of dim (Nmax,)): k(x,infty) for x=1/365,....,Nmax/365
    '''
    assert 0<=delta<=1
    if 0<delta<1:
        sqrt_D=np.sqrt(alpha**2 + 4*delta/(1-delta))
        l_1=(alpha-sqrt_D)/2
        l_2=(alpha+sqrt_D)/2
    else:
        sqrt_D=l_1=l_2=None

    arr_k_inf=np.zeros(Nmax)
    arr_x=np.arange(1,Nmax+1)/365

    if delta==0:
        arr_k_inf=-arr_x/alpha**2*np.exp(-alpha*arr_x)+\
                        2/alpha**3*(1-np.exp(-alpha*arr_x))
    elif delta==1:
        arr_k_inf=1/alpha*(1-np.exp(-alpha*arr_x))
    else:
        arr_k_inf=-alpha/(delta*l_2**2)*(1-np.exp(-l_2**arr_x))+\
            1/(alpha*delta)*(1-np.exp(-alpha*arr_x))


    return arr_k_inf

def generate_aug_kernel_matrix(alpha,
    delta,
    Nmax=3650,
    Nmax_y=None):
    '''
    - Generate augmented kernel matrix with kernel hyper-parameter alpha. The last row and column correspond to infinite maturity.
    - Args:
        - alpha (float): kernel hyper-parameter alpha
        - Nmax (int): number of rows in the output kernel matrix that correspond to finite maturity.
        - Nmax_y (int): number of columns in the output kernel matrix that correspond to finite maturity. If it is None, set Nmax_y=Nmax.
    - Returns:
        - K_aug (numpy array of dim (Nmax+1,Nmax_y+1)): kernel matrix with hyper-paramter alpha augmented with a row and a column that correspond to infinite maturity.
    '''
    if Nmax_y is None:
        Nmax_y=Nmax

    K_aug=np.full((Nmax+1,Nmax_y+1),np.nan)
    K_aug[:Nmax,:Nmax_y]=generate_kernel_matrix(alpha,delta,Nmax,Nmax_y)
    K_aug[-1,:-1]=generate_kernel_vec_inf(alpha,delta,Nmax_y)
    K_aug[:-1,-1]=generate_kernel_vec_inf(alpha,delta,Nmax)

    if delta==0:
        K_aug[-1,-1]=2/alpha**3
    elif delta==1:
        K_aug[-1,-1]=1/alpha
    else:
        sqrt_D=np.sqrt(alpha**2 + 4*delta/(1-delta))
        l_2=(alpha+sqrt_D)/2
        K_aug[-1,-1]=-alpha/(delta*l_2**2)+1/(alpha*delta)

    return K_aug



def svd_kernel_matrix(K,
                     use_randomized_svd=False,
                     max_num_components=None,
                     random_state=None):
    '''
    - Return SVD of kernel matrix K without rows and columns corresponding to infinite maturity
    - Args: 
        - K (numpy array of dim (Nmax,Nmax_y)): kernel matrix with hyper-paramter alpha. 
        - use_randomized_svd (bool): If set to True, use randomized SVD with random_date to extract up to max_num_components.
            Randomized SVD will speed up computation if K is high dimensional.
        - max_num_components (int): maximum number of components to extract in randomized SVD
        - random_state (int): random state in randomized SVD
    - Returns:
        - dict_svd (dict): dictionary containing output of SVD
    '''
    if use_randomized_svd:
        assert max_num_components is not None
        _,D_diag,Vh=randomized_svd(K, max_num_components, random_state=random_state)
    else:
        _,D_diag,Vh=np.linalg.svd(K)
        
    Vh=np.real(Vh)
    D_diag=np.real(D_diag)

    V=Vh.T
    DV_inv=V@np.diag(1/np.sqrt(D_diag))

    dict_svd={'V':V,
             'D_diag':D_diag,
             'DV_inv':DV_inv}

    return dict_svd

