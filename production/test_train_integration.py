import unittest
import tempfile
import os
import mlflow
from train import trainModel
from score import loadModel, predict, FEATURE_RANGES
from sklearn.linear_model import LogisticRegression

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.numFeatures = 2

    def test_pipelineToModel(self):
        #We're going to start saving and loading stuff here and we don't
        # want to pollute the filesystem so we'll use a temporary directory
        # that will get cleaned up when we're done.
        # Python has a nice context manager for this in the tempfile module.
        with tempfile.TemporaryDirectory() as tmpdir:
            #Use the trainmodel function to create a model
            model, trainX, trainY = trainModel('iris.csv', tmpdir + "/model")
            #Check the training data is as expected
            #Check X and Y are the same length
            self.assertEqual(len(trainX), len(trainY))
            #Check *all* the rows match what we expect
            for i in range(len(trainX)):
                #Check it has the right number of features
                self.assertEqual(len(trainX[i]), self.numFeatures)
                #Check it's one of the allowed values
                self.assertIn(trainY[i], self.classes)
                #Check if we predict it we get a allowed value from the model!
                self.assertIn(model.predict([trainX[i]])[0], self.classes)

            #Task 3: check the data it was trained on is in sensible ranges
            for row in trainX:
                for value, (name, (low, high)) in zip(row, FEATURE_RANGES.items()):
                    self.assertGreaterEqual(value, low, f"{name} too small: {value}")
                    self.assertLessEqual(value, high, f"{name} too large: {value}")

            #Check that it's the right type
            self.assertIsInstance(model, LogisticRegression)
            #Check the model matrix has the right nummber of features
            self.assertEqual(model.coef_.shape[1], self.numFeatures)
            #Check that it has stored it in the right place
            self.assertTrue(os.path.exists(tmpdir + "/model/model.pkl"))
            #Check we can load it (using the inferencing code)
            loadedModel = loadModel(tmpdir + "/model")
            #Check that it's the right type
            self.assertIsInstance(loadedModel, LogisticRegression)
            #Check the model matrix matches the saved one
            self.assertEqual(model.coef_.shape, loadedModel.coef_.shape)

            #Task 3: the loaded model makes the same predictions as the saved one
            self.assertEqual(list(model.predict(trainX)), list(loadedModel.predict(trainX)))

            #Task 3: silly values are caught by the guard-rails
            sillyInputs = [
                [9000, 3.0],   # absurd sepalLength
                [5.1, -2.0],   # negative sepalWidth
                [0, 0],        # zeros
                [5.1],         # missing a feature
            ]
            self.assertEqual(predict(loadedModel, sillyInputs), [None] * len(sillyInputs))

            #...while sensible values still get a real class back
            for result in predict(loadedModel, [[5.1, 3.5], [6.7, 3.0]]):
                self.assertIn(result, self.classes)

if __name__ == '__main__':
    unittest.main()
