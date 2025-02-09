""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    - Text columns are passed through separate LanguageModel layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""
import random, json, time, os, logging, warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, gluon
from gluoncv.utils import LRSequential, LRScheduler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, FunctionTransformer  # PowerTransformer

from ......core import Space
from ......utils import try_import_mxboard
from ....utils.loaders import load_pkl
from ..abstract.abstract_model import AbstractModel, fixedvals_from_searchspaces
from ...constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from ....metrics import log_loss, roc_auc
from .categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder
from .tabular_nn_dataset import TabularNNDataset
from .embednet import EmbedNet
from .tabular_nn_trial import tabular_nn_trial
from .hyperparameters.parameters import get_default_param
from .hyperparameters.searchspaces import get_default_searchspace

warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
logger = logging.getLogger(__name__)
EPS = 1e-10 # small number


# TODO: Gets stuck after infering feature types near infinitely in nyc-jiashenliu-515k-hotel-reviews-data-in-europe dataset, 70 GB of memory, c5.9xlarge
#  Suspect issue is coming from embeddings due to text features with extremely large categorical counts.
class TabularNeuralNetModel(AbstractModel):
    """ Class for neural network models that operate on tabular data.
        These networks use different types of input layers to process different types of data in various columns.

        Attributes:
            types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed', 'language'; values = column-names of Dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in df corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object.

        Note: This model always assumes higher values of self.eval_metric indicate better performance.

    """

    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    # TODO: remove: metric_map = {REGRESSION: 'Rsquared', BINARY: 'accuracy', MULTICLASS: 'accuracy'}  # string used to represent different evaluation metrics. metric_map[self.problem_type] produces str corresponding to metric used here.
    # TODO: should be using self.eval_metric as the metric of interest. Should have method: get_metric_name(self.eval_metric)
    rescale_losses = {gluon.loss.L1Loss:'std', gluon.loss.HuberLoss:'std', gluon.loss.L2Loss:'var'} # dict of loss names where we should rescale loss, value indicates how to rescale. Call self.loss_func.name
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # Stores temporary network parameters (eg. during the course of training)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        TabularNeuralNetModel object.

        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        eval_metric (func): function used to evaluate performance (Note: we assume higher = better)
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.types_of_features = None
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.processor = None # data processor
        self.summary_writer = None
        self.ctx = mx.cpu()
        self.batch_size = None
        self.num_dataloading_workers = None
        self.num_dataloading_workers_inference = 0
        self.params_post_fit = None
        self.num_net_outputs = None
        self._architecture_desc = None
        self.optimizer = None
        self.verbosity = None
        if self.stopping_metric is not None and self.eval_metric == roc_auc and self.stopping_metric == log_loss:
            self.stopping_metric = roc_auc  # NN is overconfident so early stopping with logloss can halt training too quick

        self.eval_metric_name = self.stopping_metric.name

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _set_default_auxiliary_params(self):
        default_auxiliary_params = dict(
            ignored_feature_types_special=['text_ngram', 'text_as_category'],
        )
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)
        super()._set_default_auxiliary_params()

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=None)

    def set_net_defaults(self, train_dataset, params):
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            self.num_net_outputs = train_dataset.num_classes
        elif self.problem_type == REGRESSION:
            self.num_net_outputs = 1
            if params['y_range'] is None:  # Infer default y-range
                y_vals = train_dataset.dataset._data[train_dataset.label_index].asnumpy()
                min_y = float(min(y_vals))
                max_y = float(max(y_vals))
                std_y = np.std(y_vals)
                y_ext = params['y_range_extend'] * std_y
                min_y = max(0, min_y-y_ext) if min_y >= 0 else min_y-y_ext
                max_y = min(0, max_y+y_ext) if max_y <= 0 else max_y+y_ext
                params['y_range'] = (min_y, max_y)
        elif self.problem_type == BINARY:
            self.num_net_outputs = 2
        else:
            raise ValueError(f"unknown problem_type specified: {self.problem_type}")

        if params['layers'] is None:  # Use default choices for MLP architecture
            if self.problem_type == REGRESSION:
                default_layer_sizes = [256, 128] # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
            else:
                default_sizes = [256, 128] # will be scaled adaptively
                # base_size = max(1, min(self.num_net_outputs, 20)/2.0) # scale layer width based on number of classes
                base_size = max(1, min(self.num_net_outputs, 100) / 50)  # TODO: Updated because it improved model quality and made training far faster
                default_layer_sizes = [defaultsize*base_size for defaultsize in default_sizes]
            # TODO: This gets really large on 100K+ rows... It takes hours on gpu for nyc-albert: 78 float/int features which get expanded to 1734, it also overfits and maxes accuracy on epoch
            #  LGBM takes 120 seconds on 4 cpu's and gets far better accuracy
            #  Perhaps we should add an order of magnitude to the pre-req with -3, or else scale based on feature count instead of row count.
            # layer_expansion_factor = np.log10(max(train_dataset.num_examples, 1000)) - 2 # scale layers based on num_training_examples
            layer_expansion_factor = 1  # TODO: Hardcoded to 1 because it results in both better model quality and far faster training time
            max_layer_width = params['max_layer_width']
            params['layers'] = [int(min(max_layer_width, layer_expansion_factor*defaultsize)) for defaultsize in default_layer_sizes]

        if train_dataset.has_vector_features() and params['numeric_embed_dim'] is None:
            # Use default choices for numeric embedding size
            vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[1]  # total dimensionality of vector features
            prop_vector_features = train_dataset.num_vector_features() / float(train_dataset.num_features) # Fraction of features that are numeric
            min_numeric_embed_dim = 32
            max_numeric_embed_dim = params['max_layer_width']
            params['numeric_embed_dim'] = int(min(max_numeric_embed_dim, max(min_numeric_embed_dim,
                                                    params['layers'][0]*prop_vector_features*np.log10(vector_dim+10) )))
        return

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, reporter=None, **kwargs):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_val (pd.DataFrame): test data features (should have same column names as Xtrain)
            y_train (pd.Series):
            y_val (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        params = self.params.copy()
        self.verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        # print('features: ', self.features)
        if 'num_cpus' in kwargs:
            self.num_dataloading_workers = max(1, int(kwargs['num_cpus']/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.batch_size = params['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X_train=X_train, y_train=y_train, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding, %d language)" %
              (train_dataset.num_examples, train_dataset.num_features,
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # self._save_preprocessor() # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.

        if 'num_gpus' in kwargs and kwargs['num_gpus'] >= 1:  # Currently cannot use >1 GPU
            self.ctx = mx.gpu()  # Currently cannot use more than 1 GPU
        else:
            self.ctx = mx.cpu()
        self.get_net(train_dataset, params=params)

        if time_limit:
            time_elapsed = time.time() - start_time
            time_limit = time_limit - time_elapsed

        self.train_net(train_dataset=train_dataset, params=params, val_dataset=val_dataset, initialize=True, setup_trainer=True, time_limit=time_limit, reporter=reporter)
        self.params_post_fit = params
        """
        # TODO: if we don't want to save intermediate network parameters, need to do something like saving in temp directory to clean up after training:
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path\
        """

    def get_net(self, train_dataset, params):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        self.set_net_defaults(train_dataset, params)
        self.model = EmbedNet(train_dataset=train_dataset, params=params, num_net_outputs=self.num_net_outputs, ctx=self.ctx)

        # TODO: Below should not occur until at time of saving
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        """ Trains neural net on given train dataset, early stops based on test_dataset.
            Args:
                train_dataset (TabularNNDataset): training data used to learn network weights
                val_dataset (TabularNNDataset): validation data used for hyperparameter tuning
                initialize (bool): set = False to continue training of a previously trained model, otherwise initializes network weights randomly
                setup_trainer (bool): set = False to reuse the same trainer from a previous training run, otherwise creates new trainer from scratch
        """
        start_time = time.time()
        logger.log(
            15,
            f"Training neural network for up to {params['num_epochs']} epochs...",
        )

        seed_value = params.get('seed_value')
        if seed_value is not None:  # Set seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            mx.random.seed(seed_value)
        if initialize:  # Initialize the weights of network
            logging.debug("initializing neural network...")
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
            logging.debug("initialized")
        if setup_trainer:
            # Also setup mxboard to monitor training if visualizer has been specified:
            visualizer = params.get('visualizer', 'none')
            if visualizer in ['tensorboard', 'mxboard']:
                try_import_mxboard()
                from mxboard import SummaryWriter
                self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=5, verbose=False)
            self.optimizer = self.setup_trainer(params=params, train_dataset=train_dataset)
        best_val_metric = -np.inf  # higher = better
        val_metric = None
        best_val_epoch = 0
        num_epochs = params['num_epochs']
        y_val = val_dataset.get_labels() if val_dataset is not None else None
        if params['loss_function'] is None:
            if self.problem_type == REGRESSION:
                params['loss_function'] = gluon.loss.L1Loss()
            elif self.problem_type == SOFTCLASS:
                params['loss_function'] = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=self.model.from_logits)
            else:
                params['loss_function'] = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)

        loss_func = params['loss_function']
        epochs_wo_improve = params['epochs_wo_improve']
        loss_scaling_factor = 1.0  # we divide loss by this quantity to stabilize gradients
        if loss_torescale := [
            key for key in self.rescale_losses if isinstance(loss_func, key)
        ]:
            loss_torescale = loss_torescale[0]
            if self.rescale_losses[loss_torescale] == 'std':
                loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS  # std-dev of labels
            elif self.rescale_losses[loss_torescale] == 'var':
                loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS  # variance of labels
            else:
                raise ValueError(
                    f"Unknown loss-rescaling type {self.rescale_losses[loss_torescale]} specified for loss_func=={loss_func}"
                )


        if self.verbosity <= 1:
            verbose_eval = -1  # Print losses every verbose epochs, Never if -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        net_filename = self.path + self.temp_file_name
        if num_epochs == 0:  # use dummy training loop that stops immediately (useful for using NN just for data preprocessing / debugging)
            logger.log(20, "Not training Neural Net since num_epochs == 0.  Neural network architecture is:")
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                if batch_idx > 0:
                    break
            self.model.save_parameters(net_filename)
            logger.log(15, "untrained Neural Net saved to file")
            return

        # Training Loop:
        for e in range(num_epochs):
            if e == 0:  # special actions during first epoch:
                logger.log(15, "Neural network architecture:")
                logger.log(15, str(self.model))  # TODO: remove?
            cumulative_loss = 0
            for data_batch in train_dataset.dataloader:
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += loss.sum()
            train_loss = cumulative_loss/float(train_dataset.num_examples)  # training loss this epoch
            if val_dataset is not None:
                # val_metric = self.evaluate_metric(test_dataset)  # Evaluate after each epoch
                val_metric = self.score(X=val_dataset, y=y_val, eval_metric=self.stopping_metric, metric_needs_y_pred=self.stopping_metric_needs_y_pred)
            if (val_dataset is None) or (val_metric >= best_val_metric) or (e == 0):  # keep training if score has improved
                if val_dataset is not None and not np.isnan(val_metric):
                    best_val_metric = val_metric
                best_val_epoch = e
                # Until functionality is added to restart training from a particular epoch, there is no point in saving params without test_dataset
                if val_dataset is not None:
                    self.model.save_parameters(net_filename)
            if val_dataset is not None:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(
                        15,
                        f"Epoch {e}.  Train loss: {train_loss.asscalar()}, Val {self.eval_metric_name}: {val_metric}",
                    )

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(tag='val_'+self.eval_metric_name,
                                                   value=val_metric, global_step=e)
            elif verbose_eval > 0 and e % verbose_eval == 0:
                logger.log(15, f"Epoch {e}.  Train loss: {train_loss.asscalar()}")
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag='train_loss', value=train_loss.asscalar(), global_step=e)  # TODO: do we want to keep mxboard support?
            if (
                reporter is not None
                and val_dataset is not None
                and (not np.isnan(val_metric))
            ):
                # epoch must be number of epochs done (starting at 1)
                reporter(epoch=e+1, validation_performance=val_metric, train_loss=float(train_loss.asscalar()))  # Higher val_metric = better
            if e - best_val_epoch > epochs_wo_improve:
                break
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.log(20, "\tRan out of time, stopping training early.")
                    break

        if val_dataset is not None:
            self.model.load_parameters(net_filename)  # Revert back to best model
            try:
                os.remove(net_filename)
            except FileNotFoundError:
                pass
        if val_dataset is None:
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)
        else: # evaluate one final time:
            final_val_metric = self.score(X=val_dataset, y=y_val, eval_metric=self.stopping_metric, metric_needs_y_pred=self.stopping_metric_needs_y_pred)
            if np.isnan(final_val_metric):
                final_val_metric = -np.inf
            logger.log(15, "Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.eval_metric_name, final_val_metric))
        self.params_trained['num_epochs'] = best_val_epoch
        return

    def _predict_proba(self, X, preprocess=True):
        """ To align predict wiht abstract_model API.
            Preprocess here only refers to feature processing stesp done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        if isinstance(X, TabularNNDataset):
            return self._predict_tabular_data(new_data=X, process=False, predict_proba=True)
        elif isinstance(X, pd.DataFrame):
            if preprocess:
                X = self.preprocess(X)
            return self._predict_tabular_data(new_data=X, process=True, predict_proba=True)
        else:
            raise ValueError(
                f"X must be of type pd.DataFrame or TabularNNDataset, not type: {type(X)}"
            )

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):    # TODO ensure API lines up with tabular.Model class.
        """ Specific TabularNN method to produce predictions on new (unprocessed) data.
            Returns 1D numpy array unless predict_proba=True and task is multi-class classification (not binary).
            Args:
                new_data (pd.Dataframe or TabularNNDataset): new data to make predictions on.
                If you want to make prediction for just a single row of new_data, pass in: new_data.iloc[[row_index]]
                process (bool): should new data be processed (if False, new_data must be TabularNNDataset)
                predict_proba (bool): should we output class-probabilities (not used for regression)
        """
        if process:
            new_data = self.process_test_data(new_data, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference, labels=None)
        if not isinstance(new_data, TabularNNDataset):
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
        if self.problem_type == REGRESSION or not predict_proba:
            preds = nd.zeros((new_data.num_examples,1))
        else:
            preds = nd.zeros((new_data.num_examples, self.num_net_outputs))
        i = 0
        for batch_idx, data_batch in enumerate(new_data.dataloader):
            data_batch = new_data.format_batch_data(data_batch, self.ctx)
            preds_batch = self.model(data_batch)
            batch_size = len(preds_batch)
            if self.problem_type != REGRESSION:
                preds_batch = (
                    nd.softmax(preds_batch, axis=1)
                    if predict_proba
                    else nd.argmax(preds_batch, axis=1, keepdims=True)
                )

            preds[i:(i+batch_size)] = preds_batch
            i = i+batch_size
        if self.problem_type == REGRESSION or not predict_proba:
            return preds.asnumpy().flatten()  # return 1D numpy array
        elif self.problem_type == BINARY:
            return preds[:,1].asnumpy()  # for binary problems, only return P(Y==+1)

        return preds.asnumpy()  # return 2D numpy array

    def generate_datasets(self, X_train, y_train, params, X_val=None, y_val=None):
        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X_train, TabularNNDataset):
            train_dataset = X_train
        else:
            X_train = self.preprocess(X_train)
            if self.features is None:
                self.features = list(X_train.columns)
            train_dataset = self.process_train_data(
                df=X_train, labels=y_train, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers,
                impute_strategy=impute_strategy, max_category_levels=max_category_levels, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features,
            )
        if X_val is None:
            val_dataset = None
        elif isinstance(X_val, TabularNNDataset):
            val_dataset = X_val
        else:
            X_val = self.preprocess(X_val)
            val_dataset = self.process_test_data(df=X_val, labels=y_val, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference)
        return train_dataset, val_dataset

    def process_test_data(self, df, batch_size, num_dataloading_workers, labels=None):
        """ Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps.
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            Dataset object
        """
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self.types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        df = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=True)

    def process_train_data(self, df, batch_size, num_dataloading_workers, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.

        # TODO no label processing for now
        # TODO: language features are ignored for now
        # TODO: how to add new features such as time features and remember to do the same for test data?
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features.
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        """
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        self.types_of_features = self._get_types_of_features(df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features) # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        df = df[self.features]
        logger.log(15, "AutoGluon Neural Network infers features are of the following types:")
        logger.log(15, json.dumps(self.types_of_features, indent=4))
        logger.log(15, "\n")
        self.processor = self._create_preprocessor(impute_strategy=impute_strategy, max_category_levels=max_category_levels)
        df = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map(max_category_levels=max_category_levels) # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map]) # should match number of columns in processed array
        # print("self.feature_arraycol_map", self.feature_arraycol_map)
        # print("num_array_cols", num_array_cols)
        # print("df.shape",df.shape)
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. Number of columns in df array does not match feature_arraycol_map.")

        # print(self.feature_arraycol_map)
        self.feature_type_map = self._get_feature_type_map() # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        # print(self.feature_type_map)
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=False)

    def setup_trainer(self, params, train_dataset=None):
        """ Set up optimizer needed for training.
            Network must first be initialized before this.
        """
        optimizer_opts = {'learning_rate': params['learning_rate'], 'wd': params['weight_decay'], 'clip_gradient': params['clip_gradient']}
        if 'lr_scheduler' in params and params['lr_scheduler'] is not None:
            if train_dataset is None:
                raise ValueError("train_dataset cannot be None when lr_scheduler is specified.")
            base_lr = params.get('base_lr', 1e-6)
            target_lr = params.get('target_lr', 1.0)
            warmup_epochs = params.get('warmup_epochs', 10)
            lr_decay = params.get('lr_decay', 0.1)
            lr_mode = params['lr_scheduler']
            num_batches = train_dataset.num_examples // params['batch_size']
            lr_decay_epoch = [max(warmup_epochs, int(params['num_epochs']/3)), max(warmup_epochs+1, int(params['num_epochs']/2)),
                              max(warmup_epochs+2, int(2*params['num_epochs']/3))]
            lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=base_lr, target_lr=target_lr, nepochs=warmup_epochs, iters_per_epoch=num_batches),
                LRScheduler(lr_mode, base_lr=target_lr, target_lr=base_lr, nepochs=params['num_epochs'] - warmup_epochs,
                            iters_per_epoch=num_batches, step_epoch=lr_decay_epoch, step_factor=lr_decay, power=2)
            ])
            optimizer_opts['lr_scheduler'] = lr_scheduler
        if params['optimizer'] == 'sgd':
            if 'momentum' in params:
                optimizer_opts['momentum'] = params['momentum']
            optimizer = gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif params['optimizer'] == 'adam':  # TODO: Can we try AdamW?
            optimizer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError(f"Unknown optimizer specified: {params['optimizer']}")
        return optimizer

    @staticmethod
    def convert_df_dtype_to_str(df):
        return df.astype(str)

    def _get_types_of_features(self, df, skew_threshold, embed_min_categories, use_ngram_features):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            Warning("Attempting to _get_types_of_features for TabularNeuralNetModel, but previously already did this.")

        feature_types = self.feature_types_metadata.feature_types_raw

        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            # print('unknown features:', unknown_features)
            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)
            # raise ValueError("unknown feature types present in DataFrame")

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'embed': [], 'language': []}
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        for feature in self.features:
            feature_data = df[feature] # pd.Series
            num_unique_vals = len(feature_data.unique())
            if num_unique_vals == 2:  # will be onehot encoded regardless of proc.embed_min_categories value
                types_of_features['onehot'].append(feature)
            elif feature in continuous_featnames:
                if np.abs(feature_data.skew()) > skew_threshold:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif feature in categorical_featnames:
                if num_unique_vals >= embed_min_categories: # sufficiently many categories to warrant learned embedding dedicated to this feature
                    types_of_features['embed'].append(feature)
                else:
                    types_of_features['onehot'].append(feature)
            elif feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features

    def _get_feature_arraycol_map(self, max_category_levels):
        """ Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature """
        feature_preserving_transforms = {'continuous', 'skewed', 'ordinal', 'language'}
        feature_arraycol_map = {} # unordered version
        current_colindex = 0
        for transformer in self.processor.transformers_:
            transformer_name = transformer[0]
            transformed_features = transformer[2]
            if transformer_name in feature_preserving_transforms:
                for feature in transformed_features:
                    if feature in feature_arraycol_map:
                        raise ValueError(
                            f"same feature is processed by two different column transformers: {feature}"
                        )

                    feature_arraycol_map[feature] = [current_colindex]
                    current_colindex += 1
            elif transformer_name == 'onehot':
                oh_encoder = [step for (name, step) in transformer[1].steps if name == 'onehot'][0]
                for i in range(len(transformed_features)):
                    feature = transformed_features[i]
                    if feature in feature_arraycol_map:
                        raise ValueError(
                            f"same feature is processed by two different column transformers: {feature}"
                        )

                    oh_dimensionality = min(len(oh_encoder.categories_[i]), max_category_levels+1)
                    # print("feature: %s, oh_dimensionality: %s" % (feature, oh_dimensionality)) # TODO! debug
                    feature_arraycol_map[feature] = list(range(current_colindex, current_colindex+oh_dimensionality))
                    current_colindex += oh_dimensionality
            else:
                raise ValueError(f"unknown transformer encountered: {transformer_name}")
        if set(feature_arraycol_map.keys()) != set(self.features):
            raise ValueError("failed to account for all features when determining column indices in processed array")
        return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])

    def _get_feature_type_map(self):
        """ Returns OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language') """
        if self.feature_arraycol_map is None:
            raise ValueError("must first call _get_feature_arraycol_map() before _get_feature_type_map()")
        vector_features = self.types_of_features['continuous'] + self.types_of_features['skewed'] + self.types_of_features['onehot']
        feature_type_map = OrderedDict()
        for feature_name in self.feature_arraycol_map:
            if feature_name in vector_features:
                feature_type_map[feature_name] = 'vector'
            elif feature_name in self.types_of_features['embed']:
                feature_type_map[feature_name] = 'embed'
            elif feature_name in self.types_of_features['language']:
                feature_type_map[feature_name] = 'language'
            else:
                raise ValueError("unknown feature type encountered")
        return feature_type_map

    def _create_preprocessor(self, impute_strategy, max_category_levels):
        """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        continuous_features = self.types_of_features['continuous']
        skewed_features = self.types_of_features['skewed']
        onehot_features = self.types_of_features['onehot']
        embed_features = self.types_of_features['embed']
        language_features = self.types_of_features['language']
        transformers = [] # order of various column transformers in this list is important!
        if len(continuous_features) > 0:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())])
            transformers.append( ('continuous', continuous_transformer, continuous_features) )
        if len(skewed_features) > 0:
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('quantile', QuantileTransformer(output_distribution='normal')) ]) # Or output_distribution = 'uniform'
                # TODO: remove old code: ('power', PowerTransformer(method=self.params['proc.power_transform_method'])) ])
            transformers.append( ('skewed', power_transformer, skewed_features) )
        if len(onehot_features) > 0:
            onehot_transformer = Pipeline(steps=[
                # TODO: Consider avoiding converting to string for improved memory efficiency
                ('to_str', FunctionTransformer(self.convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=max_category_levels, sparse=False))]) # test-time unknown values will be encoded as all zeros vector
            transformers.append( ('onehot', onehot_transformer, onehot_features) )
        if len(embed_features) > 0: # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
            ordinal_transformer = Pipeline(steps=[
                ('to_str', FunctionTransformer(self.convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=max_category_levels))]) # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            transformers.append( ('ordinal', ordinal_transformer, embed_features) )
        if len(language_features) > 0:
            raise NotImplementedError("language_features cannot be used at the moment")
        return ColumnTransformer(transformers=transformers) # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.

    def save(self, file_prefix="", directory=None, return_filename=False, verbose=True):
        """ file_prefix (str): Appended to beginning of file-name (does not affect directory in file-path).
            directory (str): if unspecified, use self.path as directory
            return_filename (bool): return the file-name corresponding to this save
        """
        if directory is not None:
            path = directory + file_prefix
        else:
            path = self.path + file_prefix

        params_filepath = path + self.params_file_name
        # TODO: Don't use os.makedirs here, have save_parameters function in tabular_nn_model that checks if local path or S3 path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.model is not None:
            self.model.save_parameters(params_filepath)
            self._architecture_desc = self.model.architecture_desc
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        modelobj_filepath = super().save(file_prefix=file_prefix, directory=directory, return_filename=True, verbose=verbose)
        self.model = temp_model
        self.summary_writer = temp_sw
        self._architecture_desc = None
        if return_filename:
            return modelobj_filepath

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        """ file_prefix (str): Appended to beginning of file-name.
            If you want to load files with given prefix, can also pass arg: path = directory+file_prefix
        """
        path = path + file_prefix
        obj: TabularNeuralNetModel = load_pkl.load(path=path + cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        if obj._architecture_desc is not None:
            obj.model = EmbedNet(architecture_desc=obj._architecture_desc, ctx=obj.ctx)  # recreate network from architecture description
            obj._architecture_desc = None
            # TODO: maybe need to initialize/hybridize??
            obj.model.load_parameters(path + cls.params_file_name, ctx=obj.ctx)
            obj.summary_writer = None
        return obj

    def hyperparameter_tune(self, X_train, y_train, X_val, y_val, scheduler_options, **kwargs):
        time_start = time.time()
        """ Performs HPO and sets self.params to best hyperparameter values """
        self.verbosity = kwargs.get('verbosity', 2)
        logger.log(15, "Beginning hyperparameter tuning for Neural Network...")
        self._set_default_searchspace() # changes non-specified default hyperparams from fixed values to search-spaces.
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        scheduler_func = scheduler_options[0]
        scheduler_options = scheduler_options[1]
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        num_cpus = scheduler_options['resource']['num_cpus']
        # num_gpus = scheduler_options['resource']['num_gpus']  # TODO: Currently unused

        params_copy = self.params.copy()

        self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        self.batch_size = params_copy['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X_train=X_train, y_train=y_train, params=params_copy, X_val=X_val, y_val=y_val)
        train_path = self.path + "train"
        val_path = self.path + "validation"
        train_dataset.save(file_prefix=train_path)
        val_dataset.save(file_prefix=val_path)

        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Warning: Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Neural Network: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, f"{str(hyperparam)}:   {str(params_copy[hyperparam])}")

        util_args = dict(
            train_path=train_path,
            val_path=val_path,
            model=self,
            time_start=time_start,
            time_limit=scheduler_options['time_out']
        )
        tabular_nn_trial.register_args(util_args=util_args, **params_copy)
        scheduler = scheduler_func(tabular_nn_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # TODO: Ensure proper working directory setup on remote machines
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading preprocessed data to remote workers...")
            scheduler.upload_files([train_path+TabularNNDataset.DATAOBJ_SUFFIX,
                                train_path+TabularNNDataset.DATAVALUES_SUFFIX,
                                val_path+TabularNNDataset.DATAOBJ_SUFFIX,
                                val_path+TabularNNDataset.DATAVALUES_SUFFIX])  # TODO: currently does not work.
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()
        scheduler.get_training_curves(plot=False, use_legend=False)

        return self._get_hpo_results(scheduler=scheduler, scheduler_options=scheduler_options, time_start=time_start)

    def get_info(self):
        info = super().get_info()
        info['hyperparameters_post_fit'] = self.params_post_fit
        return info

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None


""" General TODOs:

- Automatically decrease batch-size if memory issue arises

- Retrain final NN on full dataset (train+val). How to ensure stability here?
- OrdinalEncoder class in sklearn currently cannot handle rare categories or unknown ones at test-time, so we have created our own Encoder in category_encoders.py
There is open PR in sklearn to address this: https://github.com/scikit-learn/scikit-learn/pull/13833/files
Currently, our code uses category_encoders package (BSD license) instead: https://github.com/scikit-learn-contrib/categorical-encoding
Once PR is merged into sklearn, may want to switch: category_encoders.Ordinal -> sklearn.preprocessing.OrdinalEncoder in preprocess_train_data()

- Save preprocessed data so that we can do HPO of neural net hyperparameters more efficiently, while also doing HPO of preprocessing hyperparameters?
      Naive full HPO method requires redoing preprocessing in each trial even if we did not change preprocessing hyperparameters.
      Alternative is we save each proprocessed dataset & corresponding TabularNeuralNetModel object with its unique param names in the file. Then when we try a new HP-config, we first try loading from file if one exists.

"""
