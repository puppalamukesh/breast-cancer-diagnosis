from bcd.data import FEATURE_NAMES, describe, load_dataset, split_dataset, to_snake_case


def test_feature_names_are_snake_case():
    assert to_snake_case("mean radius") == "mean_radius"
    assert to_snake_case("worst concave points") == "worst_concave_points"
    assert len(FEATURE_NAMES) == 30
    assert all(" " not in name for name in FEATURE_NAMES)


def test_dataset_shape_and_labels():
    X, y = load_dataset()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    # 212 malignant and 357 benign in the original data
    assert int(y.sum()) == 212
    assert describe(X, y)["missing_values"] == 0


def test_split_is_stratified_and_disjoint():
    X, y = load_dataset()
    s = split_dataset(X, y)
    assert len(s.X_train) + len(s.X_test) == len(X)
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.02
