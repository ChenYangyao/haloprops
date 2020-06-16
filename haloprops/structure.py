import collections
import h5py
from sklearn import ensemble
import numpy as np
import scipy.interpolate as interp
import pkg_resources

PropType = collections.namedtuple("PropType", ['val', 'trend', 'random_noise'])

class RFRegr:
    def __init__(self, 
                 X_tr, Y_tr, n_jobs = 1, minspl=2, verbose=False, 
                 X_names = [], Y_name = '', weight=None, 
                 maxspl=30, random_state=0):
        self.oob_score = -1.0e10
        self.importance = []
        self.best_leafspl = -1
        self.best_nfeature = -1
        self.Y_tr = Y_tr + 0.
        self.Y_pred = []
        self.regressor = None
        
        ntr = 100
        leafspl_max = min( int(X_tr.shape[0] / 20), maxspl)
        feature_max = X_tr.shape[1]
        for leafspl in range(minspl, leafspl_max, 2):
            for feature in range(1, feature_max+1):
                rf = ensemble.RandomForestRegressor( 
                    ntr, min_samples_split=leafspl, max_features=feature, 
                    bootstrap=True, oob_score=True, 
                    random_state=random_state, n_jobs=n_jobs )
                rf.fit(X_tr, Y_tr, sample_weight=weight)
                if verbose: 
                    print('leaf spl=%d, nfeature=%d oob_score=%f'%(leafspl, 
                        feature,rf.oob_score_))
                if rf.oob_score_ > self.oob_score:
                    self.oob_score = rf.oob_score_
                    self.importance = rf.feature_importances_ + 0.
                    self.best_leafspl = leafspl
                    self.best_nfeature = feature
                    self.Y_pred = rf.predict( X_tr )
                    self.regressor = rf
        self.tags = {'score': self.oob_score, 
                'importance':self.importance, 
                'leaf spl': self.best_leafspl, 
                'nfeature': self.best_nfeature, 
                'Y_tr': self.Y_tr, 'Y_pred': self.Y_pred, 
                'X_names': ( tuple(range(X_tr.shape[1])) \
                    if len(X_names)==0 else X_names ), 
                'Y_name': ('Y' if Y_name == '' else Y_name)}
    def get(self):
        return self.tags
    def __getitem__(self, tag):
        return self.tags[tag]


class StructurePredictor:
    ''' predict structural parameters of dark matter halo from its assembly and 
    environment.

    Examples:
    ----------
    from haloprops.structure import StructurePredictor

    # Initialize a halo structural predictor 'sp'.
    sp = StructurePredictor()

    # Specify which structural parameter is needed. fit() will perfrom fittings 
    # of template data, which may takes some time. Here we use concentration as 
    # target.
    sp.fit('concentration')

    # Now, predict concentration from other halo parameters. Here, we provide a 
    # list of halos, each has four elements 
    #       [MAH PC1,  halo mass, tidal anisotropy, halo bias]. 
    # For the detail of these parameters, please refer to the doc string of 
    # fit() and predict() methods.
    # predict() method takes the list of halos and outputs several useful 
    # quantities, with 'val' to be the predicted structural parameter (here
    # the concentration of halos).
    X = [[ 1.55278125e+00,  2.78270723e+04,  1.88321865e-01,
        4.02548960e+00],
      [ 3.08029618e-01,  2.88465363e+02,  1.81991570e-01,
        -1.14664588e-01],
      [ 3.46576614e-01,  1.57965881e+03,  4.11933281e-01,
        2.77537420e+00]]
    concentration = sp.predict(X).val
    print(concentration)
    '''

    available_targets = ['concentration', 'shape', 'spin', 
        'shape(inner)', 'spin(inner)']
    available_predictors = ['MAH PC1', 'halo mass', 'tidal', 'bias', 
        'shape(z=3)', 'spin(z=3)', 'shape(inner)', 'shape', 'spin(inner)', 
        'spin']
    default_predictors = ['MAH PC1', 'halo mass', 'tidal', 'bias']
    _features_map = { 
        'concentration': 'c', 'shape(inner)': 'q1 6', 'shape': 'q1', 
        'spin(inner)': 'log L 6', 'spin': 'log L',
        'MAH PC1': 'MAH PCA 1',  'halo mass': 'rootmass', 
        'tidal': 'alpha r4.0', 'bias': 'bias r01', 
        'shape(z=3)': 'q1 z3', 'spin(z=3)': 'log L z3',
        'lmm': 'ftime lmm'}
    
    def __init__(self, datapath=None):
        '''initializer
        Parameters
        ----------
        @datapath:  None | str, which template data file is used. None for using 
            default template.
        '''
        self.datapath = datapath if datapath is not None \
            else self.default_datapath()
        self.regressor = None
        self.x_tr, self.y_tr = None, None
        self.interp_noise = None
        
    def fit(self, target, predictors=None, 
            method='boosting', regression_kws={},
            residual_interp_nodes=20, residual_interp_kind='slinear'):
        '''fit the template data.
        Parameters
        ----------
        @target: 'concentration' | 'shape' | 'spin' | 'shape(inner)'
            | 'spin(inner)', which structural parameter to fit. 
            Available target values can be obtained by available_targets static
            member.
            For the detail definitions see Chen Yangyao et al. 2019. The 'inner' 
            version are those calculated in 2.5 r_s where r_s is the 
            characteristic scale of NFW profile of dark matter halo.
        @predictors: None |  list-of-str, using which predictors. 
            None for default, where 'MAH PC1', 'halo mass', 'tidal' and 'bias' 
            are used. In calling predict() method you must pass these properties
            for each halo.
            Available predictors can be obtained by available_predictors static
            member.
        @method: 'boosting | 'random forest', which regressor to use in the
            template-fitting. 
        @regression_kws: additional keywords arguments passed into template-
            fitting regressor.
        @residual_interp_nodes: integer
        @residual_interp_kind: str | int, passed into the 'kind' parameter of 
            scipy.interpolate.interp1d().
            These two parameters control how to fit the residual of regressor,
            which is then used in predict() to add random noise to the mean
            trend predicted by the fitted regressor.
        '''
        if target not in self.available_targets:
            raise KeyError("target variable", target, " is not allowed")
        if predictors is None:
            predictors = self.default_predictors
        for p in predictors:
            if p not in self._features_map.keys():
                raise KeyError("predictor", p, " is not allowed")
                
        xs, y = self._load( target, predictors )        
        
        if method == "boosting":
            kws = {'learning_rate': 0.08, 'n_estimators': 1000, 
                    'validation_fraction':0.25,
                    'n_iter_no_change':10, 'random_state': 0,
                    **regression_kws}
            regr = ensemble.GradientBoostingRegressor( **kws )
            regr.fit( xs, y )
        elif method == "random forest":
            kws = {'n_jobs': None, 'minspl': 10, 'verbose': False,
                   'random_state': 0, **regression_kws}
            regr = RFRegr( xs, y, **kws ).regressor
        else:
            raise KeyError("method", method, " is invalid")
    
        self.x_tr, self.y_tr = xs, y
        self.regressor = regr
        y_pred = regr.predict(xs)
        self._build_residual(y, y_pred, 
            residual_interp_nodes, residual_interp_kind)
    
    def predict(self, predictors_value):
        '''predict halo structure parameter from other parameters. predict()
        can be called only after fit().
        @predictors_value: array-like, shape = (N, M). Here N is no. of halos
            to predict, M are the no. predictors to use.
            For each halo, M predictors must be provided, according to which
            predictors are used in fit(). Default are  M = 4 and you must 
            provide 'MAH PC1', 'halo mass', 'tidal' and 'bias'.
        '''
        if self.regressor is None:
            raise Exception("predict() must be after fit()")
        y = self.regressor.predict( predictors_value )
        sigma = self.interp_noise( y )
        dy = np.random.randn(len(y))*sigma
        return PropType(y+dy, y, dy)
    
    @staticmethod
    def default_datapath():
        '''template data file used in fitting'''
        return pkg_resources.resource_filename(
            "haloprops", "data/sample_ELUCID_L500_N3072_H10000.h5")
    
    def _load(self, target, predictors):
        _target = self._features_map[target]
        _predictors = [ self._features_map[p] for p in predictors ]
        _lmm = self._features_map['lmm']
        with h5py.File(self.datapath, 'r') as f:
            g = f['raw']
            y = g[_target][()]
            xs = np.array( [ g[p][()] for p in _predictors ] ).T
            lmm = g[_lmm][()]
        sel = (lmm > 0.3) | (lmm < -1.0e-6)
        xs = xs[sel]
        y = y[sel]
        return xs,y
    
    def _build_residual(self, y, y_pred, nodes=20, kind='slinear'):
        dy = y - y_pred
        ylo, yhi = np.quantile( y_pred, [0.01, 0.99] )
        ny = nodes
        ystep = (yhi-ylo)/ny
        ybins, yresidual = np.zeros(ny, dtype=float), np.zeros(ny, dtype=float)
        for i in range(ny):
            sel = np.array((y_pred - ylo) / ystep, dtype=int) == i
            yresidual[i] = np.std(dy[sel])
            ybins[i] = ylo + (i+0.5)*ystep
        self.interp_noise = interp.interp1d( ybins, yresidual, 
            kind=kind, fill_value = 'extrapolate')