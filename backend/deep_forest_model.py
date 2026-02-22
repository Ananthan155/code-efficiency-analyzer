"""
Deep Forest Model - Time Complexity Classifier
FIXED FOR:
✔ sklearn ≥1.4
✔ Python 3.13+
✔ CP Dataset
✔ Calibration working
✔ Cascade layers working
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import label_binarize
import pickle
import os


class DeepForestModel:

    def __init__(self, model_path='models/trained_model.pkl'):
        self.model_path = model_path
        self.complexity_classes = [
            'O(1)', 'O(log n)', 'O(n)', 'O(n log n)',
            'O(n^2)', 'O(n^3)', 'O(2^n)', 'O(n!)'
        ]
        self.n_cascades = 3
        self.cascades = []
        self.is_trained = False
        self._load_model()

    # ================= FOREST LAYER =================

    def _create_forest_layer(self, n_estimators=100):
        return [
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            ),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
        ]

    def _get_cascade_features(self, X, layer_clfs):
        probas = [clf.predict_proba(X) for clf in layer_clfs]
        return np.hstack(probas)

    def _predict_with_cascades(self, X):
        X_aug = X.copy()

        for i, (layer_clfs, _) in enumerate(self.cascades):

            if i == len(self.cascades) - 1:

                preds = np.array([clf.predict(X_aug) for clf in layer_clfs])

                final = np.apply_along_axis(
                    lambda x: np.bincount(
                        x,
                        minlength=len(self.complexity_classes)
                    ).argmax(),
                    axis=0,
                    arr=preds
                )
                return final

            else:
                proba = self._get_cascade_features(X_aug, layer_clfs)
                X_aug = np.hstack([X_aug, proba])

        return None

    def _get_calibrated_proba(self, X):

        X_aug = X.copy()

        for i, (layer_clfs, calibrated_clfs) in enumerate(self.cascades):

            if i == len(self.cascades) - 1:
                probas = [cal.predict_proba(X_aug) for cal in calibrated_clfs]
                return np.mean(probas, axis=0)

            else:
                proba = self._get_cascade_features(X_aug, layer_clfs)
                X_aug = np.hstack([X_aug, proba])

        return None

    # ================= TRAIN =================

    def train(self, training_data):

        if not training_data:
            return {'error': 'No training data'}

        X, y = [], []

        for sample in training_data:
            features = sample.get('features', {})
            complexity = sample.get('complexity', '')

            X.append(self._dict_to_vector(features))

            if complexity in self.complexity_classes:
                y.append(self.complexity_classes.index(complexity))
            else:
                y.append(2)

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"Training samples : {len(X_train)}")
        print(f"Test samples     : {len(X_test)}")

        self.cascades = []

        X_train_aug = X_train.copy()
        X_test_aug  = X_test.copy()

        for i in range(self.n_cascades):

            print(f"Training cascade layer {i+1}/{self.n_cascades}...")

            layer_clfs = self._create_forest_layer()

            for clf in layer_clfs:
                clf.fit(X_train_aug, y_train)

            # ===== SKLEARN ≥1.4 CALIBRATION FIX =====

            calibrated_clfs = []

            for clf in layer_clfs:

                cal = CalibratedClassifierCV(
                    estimator=clf,
                    method='isotonic',
                    cv=None
                )

                cal.fit(X_test_aug, y_test)
                calibrated_clfs.append(cal)

            self.cascades.append((layer_clfs, calibrated_clfs))

            if i < self.n_cascades - 1:

                train_proba = self._get_cascade_features(X_train_aug, layer_clfs)
                test_proba  = self._get_cascade_features(X_test_aug,  layer_clfs)

                X_train_aug = np.hstack([X_train_aug, train_proba])
                X_test_aug  = np.hstack([X_test_aug,  test_proba])

        y_pred   = self._predict_with_cascades(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        avg_proba = self._get_calibrated_proba(X_test)

        y_bin = label_binarize(
            y_test,
            classes=list(range(len(self.complexity_classes)))
        )

        brier = np.mean([
            brier_score_loss(y_bin[:, c], avg_proba[:, c])
            for c in range(len(self.complexity_classes))
            if y_bin[:, c].sum() > 0
        ])

        print(f"\n✔ Accuracy   : {accuracy:.4f}")
        print(f"✔ Brier Score: {brier:.4f}")

        self.is_trained = True
        self._save_model()

        return {
            'accuracy': float(accuracy),
            'brier': float(brier)
        }

    # ================= HELPERS =================

    def _dict_to_vector(self, features):

        order = [
    'num_loops',
    'num_recursive_calls',
    'num_conditionals',
    'num_function_calls',
    'num_list_operations',
    'num_dict_operations',
    'num_sorting_operations',
    'num_array_accesses',
    'code_length',
    'has_binary_search_pattern',
    'has_divide_conquer_pattern',
    'num_variables',
    'cyclomatic_complexity'
]


        return [float(features.get(f, 0)) for f in order]

    def _save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'cascades': self.cascades,
                'complexity_classes': self.complexity_classes,
                'is_trained': self.is_trained
            }, f)
        print(f"✔ Model saved")

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                self.cascades = data['cascades']
                self.complexity_classes = data['complexity_classes']
                self.is_trained = data['is_trained']
                print("✔ Model loaded")
        except:
            self.is_trained = False
