from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# EHNN model options
__C.EHNN = edict()
__C.EHNN.FEATURE_CHANNEL = 512
__C.EHNN.SK_ITER_NUM = 10
__C.EHNN.SK_EPSILON = 1e-10
__C.EHNN.SK_TAU = 0.005
__C.EHNN.MGM_SK_TAU = 0.005
__C.EHNN.GNN_FEAT = [16, 16, 16]
__C.EHNN.GNN_LAYER = 3
__C.EHNN.GAUSSIAN_SIGMA = 1.
__C.EHNN.SIGMA3 = 1.
__C.EHNN.WEIGHT2 = 1.
__C.EHNN.WEIGHT3 = 1.
__C.EHNN.TRANSFORMER = False
__C.EHNN.N_HEADS = 1
__C.EHNN.EDGE_FEATURE = 'cat'  # 'cat' or 'geo'
__C.EHNN.ORDER3_FEATURE = 'none'  # 'cat' or 'geo' or 'none'
__C.EHNN.FIRST_ORDER = True
__C.EHNN.EDGE_EMB = False
__C.EHNN.SK_EMB = 1
__C.EHNN.GUMBEL_SK = 0  # 0 for no gumbel, other wise for number of gumbel samples
__C.EHNN.UNIV_SIZE = -1
__C.EHNN.POSITIVE_EDGES = True
__C.EHNN.FFN_DROPOUT = 0.
__C.EHNN.ATT0_DROPOUT = 0.
__C.EHNN.ATT1_DROPOUT = 0.
