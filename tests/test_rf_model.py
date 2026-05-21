import numpy as np
from src.rf_model import train_random_forest, predict, save_rf, load_rf
from src.utils import NUM_CLASSES
import os
import tempfile


class TestRandomForest:
    def test_train_and_predict(self):
        n_samples = 100
        feature_dim = 128
        X = np.random.randn(n_samples, feature_dim).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=n_samples)
        weights = np.random.rand(n_samples).astype(float)

        rf = train_random_forest(X, y, sample_weights=weights, n_estimators=20)
        results = predict(rf, X[:5])

        assert len(results) == 5
        for r in results:
            assert "class_code" in r
            assert "class_name" in r
            assert "confidence" in r
            assert 0 <= r["confidence"] <= 1

    def test_save_and_load(self):
        X = np.random.randn(50, 64).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=50)
        rf = train_random_forest(X, y, n_estimators=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rf.joblib")
            save_rf(rf, path)
            loaded = load_rf(path)
            assert loaded is not None
            pred_orig = rf.predict(X[:3])
            pred_loaded = loaded.predict(X[:3])
            assert np.array_equal(pred_orig, pred_loaded)
