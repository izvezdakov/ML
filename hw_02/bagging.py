import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        """
        Generate indices for every bag and store in self.indices_list list
        """
        self.indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            # bootstrapped sample: choose `data_length` indices WITH replacement,
            # so some objects are duplicated and some are missed
            bag_indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(bag_indices)

    def fit(self, model_constructor, data, target):
        """
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        """
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(map(len, self.indices_list))) == 1, 'All bags should be of the same length!'
        assert len(self.indices_list[0]) == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            bag_indices = self.indices_list[bag]
            data_bag = data[bag_indices]
            target_bag = target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        """
        Get average prediction for every object from passed dataset
        """
        predictions = np.array([model.predict(data) for model in self.models_list])
        return predictions.mean(axis=0)

    def _get_oob_predictions_from_every_model(self):
        """
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        """
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for model, bag_indices in zip(self.models_list, self.indices_list):
            # mark which objects were NOT used to train this model
            is_out_of_bag = np.ones(len(self.data), dtype=bool)
            is_out_of_bag[bag_indices] = False
            oob_positions = np.nonzero(is_out_of_bag)[0]
            if len(oob_positions) == 0:
                continue
            oob_predictions = model.predict(self.data[oob_positions])
            for position, prediction in zip(oob_positions, oob_predictions):
                list_of_predictions_lists[position].append(prediction)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        """
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        """
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array(
            [np.mean(preds) if preds else None for preds in self.list_of_predictions_lists],
            dtype=object,
        )

    def OOB_score(self):
        """
        Compute mean square error for all objects, which have at least one prediction
        """
        self._get_averaged_oob_predictions()
        known_positions = [
            i for i, prediction in enumerate(self.oob_predictions) if prediction is not None
        ]
        if not known_positions:
            return None
        y_true = self.target[known_positions]
        y_pred = np.array([self.oob_predictions[i] for i in known_positions])
        return np.mean((y_true - y_pred) ** 2)
