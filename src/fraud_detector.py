"""
Fraud detection using a RandomForest trained on synthetic documents.

The saved model currently uses the image-derived feature vector.
OCR and QR information are used by the validation pipeline, while
the fraud model receives the same feature representation used during
training.
"""

import os
import pickle

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.features import (
    extract_features,
    FEATURE_NAMES,
)

from src.dataset_generator import (
    generate_dataset,
    label_for_path,
)

from src.preprocessing import load_image


RISK_CLASSES = [
    "LOW",
    "MEDIUM",
    "HIGH",
]


LABEL_TO_RISK = {
    "GENUINE": "LOW",
    "MANIPULATED": "HIGH",
}


def _build_dataset(
    dataset_dir: str
) -> tuple:

    X = []
    y = []

    for root, _, files in os.walk(
        dataset_dir
    ):

        for filename in files:

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            try:

                image = load_image(
                    full_path
                )

                # IMPORTANT:
                # Training uses image-only features.
                features = extract_features(
                    image
                )

                X.append([
                    features[name]
                    for name in FEATURE_NAMES
                ])

                raw_label = (
                    label_for_path(
                        full_path
                    )
                )

                risk_label = (
                    LABEL_TO_RISK.get(
                        raw_label,
                        "MEDIUM"
                    )
                )

                y.append(
                    risk_label
                )

            except Exception as error:

                print(
                    f"[!] Skipping "
                    f"{full_path}: "
                    f"{error}"
                )

    return (
        np.array(X),
        np.array(y)
    )


def _synthesize_medium_examples(
    X_low: np.ndarray,
    X_high: np.ndarray,
    n: int = 20,
    random_state: int = 42
) -> np.ndarray:

    rng = np.random.default_rng(
        random_state
    )

    output = []

    for _ in range(n):

        i = rng.integers(
            0,
            len(X_low)
        )

        j = rng.integers(
            0,
            len(X_high)
        )

        alpha = rng.uniform(
            0.3,
            0.7
        )

        mixed = (
            X_low[i] * alpha
            + X_high[j] * (1 - alpha)
        )

        output.append(
            mixed
        )

    return np.array(
        output
    )


def train_fraud_model(
    dataset_dir: str,
    model_output_path: str,
    n_genuine: int = 40,
    n_manipulated: int = 40,
    n_medium: int = 20,
    random_state: int = 42
) -> tuple:

    print(
        f"Generating synthetic dataset "
        f"in {dataset_dir}..."
    )

    generate_dataset(
        dataset_dir,
        n_genuine=n_genuine,
        n_manipulated=n_manipulated
    )

    X, y = _build_dataset(
        dataset_dir
    )

    # Reinforce genuine multi-document references (Aadhaar, PAN, DL)
    samples_dir = os.path.join(os.path.dirname(dataset_dir), "samples")
    if os.path.isdir(samples_dir):
        from src.dataset_generator import _manipulate_document
        ref_files = [
            "sample_aadhaar.jpg",
            "sample_pan.jpg",
            "sample_license.jpg",
            "card_333x248.png",
        ]
        extra_X = []
        extra_y = []
        for rname in ref_files:
            rpath = os.path.join(samples_dir, rname)
            if os.path.exists(rpath):
                ref_img = load_image(rpath)
                feats = extract_features(ref_img)
                for _ in range(8):
                    extra_X.append([feats[name] for name in FEATURE_NAMES])
                    extra_y.append("LOW")
                for _ in range(8):
                    m_img = _manipulate_document(ref_img)
                    m_feats = extract_features(m_img)
                    extra_X.append([m_feats[name] for name in FEATURE_NAMES])
                    extra_y.append("HIGH")
        if extra_X:
            X = np.vstack([X, np.array(extra_X)])
            y = np.concatenate([y, np.array(extra_y)])

    print(
        f"Dataset size: {len(X)}"
    )

    X_low = X[
        y == "LOW"
    ]

    X_high = X[
        y == "HIGH"
    ]

    if (
        len(X_low) > 0
        and len(X_high) > 0
    ):

        X_medium = (
            _synthesize_medium_examples(
                X_low,
                X_high,
                n=n_medium,
                random_state=random_state,
            )
        )

        y_medium = np.array(
            ["MEDIUM"]
            * len(X_medium)
        )

        X = np.vstack(
            [X, X_medium]
        )

        y = np.concatenate(
            [y, y_medium]
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=random_state,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    present_labels = sorted(
        set(y_test)
        | set(predictions),
        key=lambda value:
            RISK_CLASSES.index(value)
            if value in RISK_CLASSES
            else 99
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=present_labels
    ).tolist()

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        classes = list(
            model.classes_
        )

        if "HIGH" in classes:

            high_index = (
                classes.index(
                    "HIGH"
                )
            )

            high_binary = (
                y_test == "HIGH"
            ).astype(int)

            auc = float(
                roc_auc_score(
                    high_binary,
                    probabilities[
                        :, high_index
                    ]
                )
            )

        else:

            auc = None

    except Exception:

        auc = None

    importances = {

        FEATURE_NAMES[index]:
            float(
                model.feature_importances_[
                    index
                ]
            )

        for index in range(
            len(FEATURE_NAMES)
        )
    }

    directory = os.path.dirname(
        model_output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        model_output_path,
        "wb"
    ) as file:

        pickle.dump(
            {
                "model": model,
                "feature_names":
                    FEATURE_NAMES,
            },
            file
        )

    summary = {

        "n_samples":
            int(len(X)),

        "n_train":
            int(len(X_train)),

        "n_test":
            int(len(X_test)),

        "classification_report":
            report,

        "confusion_matrix":
            matrix,

        "confusion_matrix_labels":
            present_labels,

        "roc_auc_high_class":
            auc,

        "feature_importances":
            importances,

        "model_path":
            os.path.abspath(
                model_output_path
            ),
    }

    return summary, model


def load_fraud_model(
    model_path: str
):

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}"
        )

    with open(
        model_path,
        "rb"
    ) as file:

        bundle = pickle.load(
            file
        )

    return (
        bundle["model"],
        bundle["feature_names"],
    )


def predict_fraud_risk(
    image: np.ndarray,
    model,
    feature_names: list,
    ocr_result: dict = None,
    qr_result: dict = None
) -> dict:

    # IMPORTANT:
    # The existing model was trained using
    # image-only features, so inference must use
    # exactly the same representation.
    features = extract_features(
        image
    )

    x = np.array([
        [
            features[name]
            for name in feature_names
        ]
    ])

    prediction = model.predict(
        x
    )[0]

    probabilities = (
        model.predict_proba(x)[0]
    )

    classes = list(
        model.classes_
    )

    probability_dict = {
        str(classes[index]):
            float(probabilities[index])

        for index in range(
            len(classes)
        )
    }

    return {

        "predicted_risk":
            str(prediction),

        "class_probabilities":
            probability_dict,

        "features_used":
            features,
    }


def explain_with_shap(
    model,
    feature_names: list,
    background_features: np.ndarray,
    instance_features: np.ndarray,
    top_k: int = 5
) -> list:

    try:

        import shap

    except ImportError:

        return []

    try:

        explainer = (
            shap.TreeExplainer(
                model
            )
        )

        shap_values = (
            explainer.shap_values(
                instance_features
            )
        )

        if isinstance(
            shap_values,
            list
        ):

            probabilities = (
                model.predict_proba(
                    instance_features
                )[0]
            )

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            values = (
                shap_values[
                    predicted_index
                ][0]
            )

        else:

            values = np.asarray(
                shap_values
            )

            # Handle newer SHAP output shape
            if values.ndim == 3:

                probabilities = (
                    model.predict_proba(
                        instance_features
                    )[0]
                )

                predicted_index = int(
                    np.argmax(
                        probabilities
                    )
                )

                values = values[
                    0,
                    :,
                    predicted_index
                ]

            elif values.ndim == 2:

                values = values[0]

            else:

                values = values.flatten()

        values = np.asarray(
            values
        ).flatten()

        pairs = []

        for index, value in enumerate(
            values
        ):

            if index >= len(
                feature_names
            ):
                break

            pairs.append(
                (
                    feature_names[index],
                    float(value)
                )
            )

        pairs.sort(
            key=lambda pair:
                abs(pair[1]),
            reverse=True
        )

        return pairs[:top_k]

    except Exception as error:

        print(
            f"[!] SHAP explanation "
            f"failed: {error}"
        )

        return []