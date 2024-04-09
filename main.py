import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge

## Abstract classes (interfaces) for predictors and transformers
class PredictorModel(ABC):
    """Predictor model."""

    @abstractmethod
    def predict(feature_df):
        """
        Predict target values.

        Parameters
        ----------
        feature_df : pd.DataFrame

        Returns
        -------
        predict_df : pd.DataFrame
        """
        pass


class PredictorLearner(ABC):
    """Predictor learner."""

    @abstractmethod
    def learn(feature_df, target_df):
        """
        Generates predictor model by using training data.

        Parameters
        ----------
        feature_df : pd.DataFrame
        target_df : pd.DataFrame

        Returns
        -------
        predictor_model : PredictorModel
        """
        pass


class TransformerModel(ABC):
    """Transformer model."""

    @abstractmethod
    def transform(feature_df):
        """
        Transforms the feature values.

        Parameters
        ----------
        feature_df : pd.DataFrame

        Returns
        -------
        trans_df : pd.DataFrame
        """
        pass


class TransformerLearner(ABC):
    """Transformer learner."""

    @abstractmethod
    def learn(feature_df):
        """
        Generates transformer model by using training data.

        Parameters
        ----------
        feature_df : pd.DataFrame

        Returns
        -------
        transformer_model : TransformerModel
        """
        pass

## Pipeline model class
class PipelineModel(object):
    """Pipeline model.

    Attributes
    ----------
    transformer_models : list[TransformerModel]
    predictor_model : PredictorModel

    """

    def __init__(self, transformer_models, predictor_model):
        self.transformer_models = transformer_models
        self.predictor_model = predictor_model

    def predict(self, feature_df):
        """
        Predict target values by transforming feature values and prediction.

        Parameters
        ----------
        feature_df : pd.DataFrame

        Returns
        -------
        predict_df : pd.DataFrame
        """
        trans_df = feature_df
        for model in self.transformer_models:
            trans_df = model.transform(trans_df)

        return self.predictor_model.predict(trans_df)

## Exercise 1.Please implement the `StandardScaler.transform(...)` and `StandardScalerLearner.learn(...)`
# which transforms feature matrix so that the average and standard deviation of each column will be 0.0 and 1.0, respectively.
class ReplaceNanWithZeroTransformer(TransformerModel):
    """Transformer to replace NaN in the feature values to zeros."""

    def transform(self, feature_df):
        trans_df = feature_df.copy()
        trans_df[feature_df.isna()] = 0.0
        return trans_df

class StandardScaler(TransformerModel):
    """Standard scaler.

    Attributes
    ----------
    mean : pd.Series
        Mean value for each column.
    stdev : pd.Series
        Standard deviation value for each column.
    """
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def transform(self, feature_df):
        return (feature_df - self.mean) / self.stdev

class StandardScalerLearner(TransformerLearner):
    """Standard scaler learner."""
    def learn(self, feature_df):
        # Calculating the mean and standard deviation for each column
        mean = feature_df.mean()
        stdev = feature_df.std()
        return StandardScaler(mean, stdev)

def test_standard_scaler():
    feature_tr_df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [-2.0, 2.0, 0.0]})
    feature_te_df = pd.DataFrame({"col1": [1.0, 0.5, 3.2], "col2": [4.0, 1.2, 0.3]})

    output_df = StandardScalerLearner().learn(feature_tr_df).transform(feature_te_df)

    expected_df = pd.DataFrame({"col1": [-1.0, -1.5, 1.2], "col2": [2.0, 0.6, 0.15]})

    try:
        assert_frame_equal(output_df, expected_df)
        print("Exercise 1 Passed")
    except AssertionError as e:
        print("Failed:", e)

test_standard_scaler()

# Exercise 2. Please implement the `PipelineLearner.learn(...)`. The test written below has two elements for `transformers`
# but it could be different number, in general.
class RidgeRegressionModel(PredictorModel):
    def __init__(self, ridge, columns):
        self.ridge = ridge
        self.columns = columns

    def predict(self, feature_df):
        return pd.DataFrame(self.ridge.predict(feature_df.values), columns=self.columns)


class RidgeRegressionLearner(PredictorLearner):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def learn(self, feature_df, target_df):
        ridge = Ridge(self.alpha)
        ridge.fit(feature_df.values, target_df.values)
        return RidgeRegressionModel(ridge, target_df.columns)

class PipelineLearner(object):
    """Pipeline learner.

    Attributes
    ----------
    transformers : list[TransformerModel or TransformerLearner]
    predictor_learner : PredictorLearner
    """

    def __init__(self, transformers, predictor_learner):
        self.transformers = transformers
        self.predictor_learner = predictor_learner

    def learn(self, feature_df, target_df):
        # Initialize an empty list to store the transformer models
        transformer_models = []

        # Apply each transformer learner to the features in sequence
        for transformer in self.transformers:
            # Check if the transformer is a model or a learner
            if isinstance(transformer, TransformerModel):
                # If it's a model, just transform the features
                feature_df = transformer.transform(feature_df)
                transformer_models.append(transformer)
            else:
                # If it's a learner, first learn the model then transform the features
                model = transformer.learn(feature_df)
                feature_df = model.transform(feature_df)
                transformer_models.append(model)

        # Train the predictor learner with the final set of transformed features and the target
        predictor_model = self.predictor_learner.learn(feature_df, target_df)

        # Return a pipeline model containing all the transformer models and the predictor model
        return PipelineModel(transformer_models, predictor_model)

def test_learner_pipeline():
    # define ML pipeline
    pipeline_learner = PipelineLearner(
        transformers=[
            ReplaceNanWithZeroTransformer(),
            StandardScalerLearner()
        ],
        predictor_learner=RidgeRegressionLearner(alpha=0.5)
    )

    # make ML pipeline model
    feature_tr_df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [-2.0, 2.0, 0.0]})
    feature_te_df = pd.DataFrame({"col1": [1.0, 0.5, 3.2], "col2": [4.0, 1.2, 0.3]})
    target_tr_df = pd.DataFrame({"target": [1.0, 2.0, -1.0]})

    pipeline_model = pipeline_learner.learn(feature_tr_df, target_tr_df)

    # predict target value by the ML pipeline model
    predict_df = pipeline_model.predict(feature_te_df)

    expected_df = pd.DataFrame({"target": [3.523810, 2.895238, -0.576190]})
    try:
        assert_frame_equal(expected_df, predict_df)
        print("Exercise 2 Passed")
    except AssertionError as e:
        print("Failed:", e)

test_learner_pipeline()

# Exercise 3. In the above cell, even if `StandardScalerLearner` uses not the output of the previous step but the original features unexpectedly, the test passes.
# Please add a new test case `test_learner_pipeline2()` for testing whether the step (transformer) correctly uses the output of the previous step as the input.
def test_learner_pipeline2():
    # define ML pipeline with a logging transformer to track transformations
    pipeline_learner = PipelineLearner(
        transformers=[
            ReplaceNanWithZeroTransformer(),
            StandardScalerLearner()
        ],
        predictor_learner=RidgeRegressionLearner(alpha=0.5)
    )

    # Use the data of Exercise2 for the test, but change the 0.0 to np.nan.
    # It can test if ReplaceNanWithZeroTransformer worked or not.
    feature_tr_df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [-2.0, 2.0, np.nan]})
    feature_te_df = pd.DataFrame({"col1": [1.0, 0.5, 3.2], "col2": [4.0, 1.2, 0.3]})
    target_tr_df = pd.DataFrame({"target": [1.0, 2.0, -1.0]})

    # Train the pipeline
    pipeline_model = pipeline_learner.learn(feature_tr_df, target_tr_df)

    # Predict using the trained pipeline model
    predict_df = pipeline_model.predict(feature_te_df)

    # Check if the marker column is correctly added and incremented
    expected_df = pd.DataFrame({"target": [3.523810, 2.895238, -0.576190]})
    try:
        assert_frame_equal(expected_df, predict_df)
        print("Exercise 3 Passed")
    except AssertionError as e:
        print("Failed:", e)

test_learner_pipeline2()
