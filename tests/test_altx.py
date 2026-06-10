"""Unit tests for the Altx class."""

import os
import tempfile

import numpy as np
import pytest
import torch

from altx import ALT as Altx  # noqa: N811

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (train_set, train_classes): 4 instances x 20 time steps, 2 classes."""
    torch.manual_seed(0)
    return torch.randn(4, 20), torch.tensor([0, 0, 1, 1])


@pytest.fixture
def model(train_data: tuple[torch.Tensor, torch.Tensor]) -> Altx:
    """Return an untrained Altx model with L=3, K=1 (R defaults to 5)."""
    data, classes = train_data
    return Altx(data, classes, L=3, K=1)


@pytest.fixture
def trained_model(model: Altx) -> Altx:
    """Return a trained Altx model."""
    model.train()
    return model


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for Altx.__init__ parameter validation and setup."""

    def test_basic_construction_from_tensors(self, train_data: tuple) -> None:
        """Valid tensor inputs produce a model with correct attributes."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=1)
        assert alt.tau == 4
        assert alt.m == 1
        assert alt.noc == 2

    def test_construction_from_numpy_arrays(self) -> None:
        """NumPy arrays are converted to tensors without error."""
        data = np.random.default_rng(0).standard_normal((4, 20)).astype(np.float32)
        classes = np.array([0, 0, 1, 1])
        alt = Altx(data, classes, L=3, K=1)
        assert isinstance(alt.train_set, torch.Tensor)
        assert isinstance(alt.train_classes, torch.Tensor)

    def test_2d_train_set_is_unsqueezed_to_3d(self, train_data: tuple) -> None:
        """A 2-D (instances, time) train_set is promoted to (instances, 1, time)."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=1)
        assert alt.train_set.ndim == 3
        assert alt.train_set.shape == (4, 1, 20)

    def test_3d_train_set_preserved(self) -> None:
        """A 3-D (instances, channels, time) train_set is kept as-is."""
        torch.manual_seed(0)
        data = torch.randn(4, 2, 20)
        classes = torch.tensor([0, 0, 1, 1])
        alt = Altx(data, classes, L=3, K=1)
        assert alt.train_set.shape == (4, 2, 20)
        assert alt.m == 2

    def test_mismatched_lengths_raises(self, train_data: tuple) -> None:
        """train_set and train_classes with different row counts raise ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="same length along the first"):
            Altx(data, classes[:3], L=3, K=1)

    def test_train_length_2d_raises(self, train_data: tuple) -> None:
        """A 2-D train_length tensor raises ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="1 dimension"):
            Altx(data, classes, train_length=torch.full((4, 2), 20), L=3, K=1)

    def test_train_length_wrong_size_raises(self, train_data: tuple) -> None:
        """train_length with wrong number of elements raises ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="same dimension"):
            Altx(
                data,
                classes,
                train_length=torch.full((3,), 20, dtype=torch.int),
                L=3,
                K=1,
            )

    def test_train_length_valid(self, train_data: tuple) -> None:
        """A correctly shaped train_length is accepted."""
        data, classes = train_data
        lengths = torch.full((4,), 20, dtype=torch.int)
        alt = Altx(data, classes, train_length=lengths, L=3, K=1)
        assert torch.all(alt.train_length == 20)

    def test_k_int_l_int(self, train_data: tuple) -> None:
        """K=int, L=int produces a single RLK triplet."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=1)
        assert len(alt.RLK) == 1

    def test_k_int_l_list(self, train_data: tuple) -> None:
        """K=int, L=list broadcasts K to match L length."""
        data, classes = train_data
        alt = Altx(data, classes, L=[3, 4], K=1)
        assert len(alt.RLK) == 2
        assert all(rlk[2] == 1 for rlk in alt.RLK)

    def test_k_list_l_int(self, train_data: tuple) -> None:
        """K=list, L=int broadcasts L to match K length."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=[1, 2])
        assert len(alt.RLK) == 2
        assert all(rlk[1] == 3 for rlk in alt.RLK)

    def test_k_list_l_list_same_length(self, train_data: tuple) -> None:
        """K=list and L=list of equal length is accepted."""
        data, classes = train_data
        alt = Altx(data, classes, L=[3, 4], K=[1, 2])
        assert len(alt.RLK) == 2

    def test_k_list_l_list_different_lengths_raises(self, train_data: tuple) -> None:
        """K and L lists of different lengths raise TypeError."""
        data, classes = train_data
        with pytest.raises(TypeError, match="two lists with"):
            Altx(data, classes, L=[3, 4], K=[1, 2, 3])

    def test_r_none_defaults_to_2l_minus_1(self, train_data: tuple) -> None:
        """R=None defaults to 2*L-1 for each L."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=1, R=None)
        assert alt.RLK[0][0] == 5  # 2*3-1 = 5

    def test_r_int_is_broadcast(self, train_data: tuple) -> None:
        """R=int is broadcast to all L values."""
        data, classes = train_data
        alt = Altx(data, classes, L=[3, 3], K=1, R=5)
        assert all(rlk[0] == 5 for rlk in alt.RLK)

    def test_r_list(self, train_data: tuple) -> None:
        """R=list of matching length is accepted."""
        data, classes = train_data
        alt = Altx(data, classes, L=[3, 4], K=1, R=[5, 7])
        assert alt.RLK[0][0] == 5
        assert alt.RLK[1][0] == 7

    def test_r_wrong_type_raises(self, train_data: tuple) -> None:
        """R of an unsupported type raises TypeError."""
        data, classes = train_data
        with pytest.raises(TypeError, match="None, int or list"):
            Altx(data, classes, L=3, K=1, R=5.0)  # type: ignore[arg-type]

    def test_r_list_wrong_length_raises(self, train_data: tuple) -> None:
        """R=list with length != len(L) raises ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="same length"):
            Altx(data, classes, L=[3, 4], K=1, R=[5])

    def test_r_too_large_raises(self, train_data: tuple) -> None:
        """R larger than the shortest training instance raises ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="too large"):
            Altx(data, classes, L=3, K=1, R=21)

    def test_r_wrong_form_raises(self, train_data: tuple) -> None:
        """R that does not satisfy step*(2*L-2)+1 raises ValueError."""
        data, classes = train_data
        # L=3: valid R values are 5, 9, 13, ... (step*(2*3-2)+1 = 4k+1)
        # R=4 is invalid: (4-1) % (3*2-2) = 3 % 4 = 3 != 0
        with pytest.raises(ValueError, match="step\\*\\(2\\*L-2\\)\\+1"):
            Altx(data, classes, L=3, K=1, R=4)

    def test_negative_l_raises(self, train_data: tuple) -> None:
        """Negative L raises ValueError."""
        data, classes = train_data
        with pytest.raises(ValueError, match="positive"):
            Altx(data, classes, L=-1, K=1)

    def test_rlk_tuple_structure(self, train_data: tuple) -> None:
        """RLK is a tuple of (r, l, k) triples matching the given parameters."""
        data, classes = train_data
        alt = Altx(data, classes, L=[3, 4], K=[1, 2], R=[5, 7])
        assert alt.RLK == ((5, 3, 1), (7, 4, 2))

    def test_class_labels_and_noc(self, train_data: tuple) -> None:
        """noc and class_labels reflect unique classes in train_classes."""
        data, classes = train_data
        alt = Altx(data, classes, L=3, K=1)
        assert alt.noc == 2
        assert set(alt.class_labels.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# _embed
# ---------------------------------------------------------------------------


class TestEmbed:
    """Tests for Altx._embed."""

    def test_returns_l_by_l_matrix(self, model: Altx) -> None:
        """Embedded matrix has shape (l, l) for the chosen l."""
        _, l, _ = model.RLK[0]
        S = model._embed(0, 0, model.RLK[0], t=0)
        assert S.shape == (l, l)

    def test_is_symmetric(self, model: Altx) -> None:
        """Embedded Hankel matrix is symmetric."""
        S = model._embed(0, 0, model.RLK[0], t=0)
        assert torch.allclose(S, S.T, atol=1e-6)

    def test_different_t_gives_different_matrix(self, model: Altx) -> None:
        """Different starting times produce different embedded matrices."""
        S0 = model._embed(0, 0, model.RLK[0], t=0)
        S1 = model._embed(0, 0, model.RLK[0], t=1)
        assert not torch.equal(S0, S1)


# ---------------------------------------------------------------------------
# _get_law
# ---------------------------------------------------------------------------


class TestGetLaw:
    """Tests for Altx._get_law."""

    def test_returns_vector_of_length_l(self, model: Altx) -> None:
        """Result is a 1-D tensor whose length equals l."""
        _, l, _ = model.RLK[0]
        S = model._embed(0, 0, model.RLK[0], t=0)
        law = model._get_law(S)
        assert law.ndim == 1
        assert law.shape[0] == l

    def test_result_is_unit_vector(self, model: Altx) -> None:
        """eigh returns orthonormal eigenvectors, so the law has unit norm."""
        S = model._embed(0, 0, model.RLK[0], t=0)
        law = model._get_law(S)
        assert torch.isclose(torch.linalg.norm(law), torch.tensor(1.0), atol=1e-5)

    def test_known_diagonal_matrix(self, model: Altx) -> None:
        """Eigenvector of diag(1, 4, 9) for eigenvalue 1 is the first basis vector."""
        D = torch.diag(torch.tensor([1.0, 4.0, 9.0]))
        law = model._get_law(D)
        # Smallest eigenvalue is 1 → eigenvector is ±e_0 = [±1, 0, 0]
        assert torch.allclose(law.abs(), torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# _nol
# ---------------------------------------------------------------------------


class TestNol:
    """Tests for Altx._nol."""

    def test_formula(self, model: Altx) -> None:
        """_nol matches the formula (t-r+1)//k + 1."""
        r, _, k = model.RLK[0]
        t = model.train_set.shape[2]
        expected = (t - r + 1) // k + 1
        assert model._nol(r, k) == expected

    def test_upper_bounds_actual_window_count(self, model: Altx) -> None:
        """_nol is >= the actual number of windows generated per instance."""
        r, _, k = model.RLK[0]
        t_len = model.train_set.shape[2]
        actual = len(range(0, t_len - r + 1, k))
        assert model._nol(r, k) >= actual


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    """Tests for Altx.train."""

    def test_train_populates_ps_for_all_rlk(self, model: Altx) -> None:
        """After training, Ps contains an entry for every RLK triplet."""
        model.train()
        for rlk in model.RLK:
            assert rlk in model.Ps

    def test_ps_shapes_consistent(self, model: Altx) -> None:
        """P_classes length equals the number of columns in P."""
        model.train()
        for rlk in model.RLK:
            p_classes, p = model.Ps[rlk]
            assert p_classes.shape[0] == p.shape[1]

    def test_ps_column_dimension_equals_l(self, model: Altx) -> None:
        """P has shape (l, n_laws, m) after training."""
        model.train()
        for rlk in model.RLK:
            _, l, _ = rlk
            _, p = model.Ps[rlk]
            assert p.shape[0] == l
            assert p.shape[2] == model.m

    def test_p_classes_values_match_train_classes(self, model: Altx) -> None:
        """All class labels in P_classes are a subset of the training classes."""
        model.train()
        for rlk in model.RLK:
            p_classes, _ = model.Ps[rlk]
            allowed = set(model.train_classes.tolist())
            assert set(p_classes.tolist()).issubset(allowed)

    def test_cleanup_removes_training_data(self, model: Altx) -> None:
        """cleanup=True deletes train_set, train_classes, tau, and train_length."""
        model.train(cleanup=True)
        assert not hasattr(model, "train_set")
        assert not hasattr(model, "train_classes")
        assert not hasattr(model, "tau")
        assert not hasattr(model, "train_length")

    def test_train_after_cleanup_raises(self, model: Altx) -> None:
        """Calling train() after cleanup raises because train_set is gone."""
        model.train(cleanup=True)
        # AttributeError because train_set was deleted (not set to None)
        with pytest.raises((AttributeError, RuntimeError)):
            model.train()


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for Altx.save and Altx.load."""

    def test_save_load_preserves_attributes(self, trained_model: Altx) -> None:
        """Loaded model has the same RLK, m, noc, and class_labels."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trained_model.save(path)
            loaded = Altx.load(path)
            assert loaded.RLK == trained_model.RLK
            assert loaded.m == trained_model.m
            assert loaded.noc == trained_model.noc
            assert torch.equal(loaded.class_labels, trained_model.class_labels)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_transform(self, trained_model: Altx) -> None:
        """Loaded model produces identical transform output for the same instance.

        Notes
        -----
        ``load()`` does not persist ``device``, so it must be set manually
        before calling ``transform``.  This is a known limitation of the
        current implementation.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trained_model.save(path)
            loaded = Altx.load(path)
            # device is not saved — restore it before transform
            loaded.device = trained_model.device  # type: ignore[attr-defined]
            torch.manual_seed(42)
            z = torch.randn(20)
            extr_methods_orig = [["mean", 0.05]]
            extr_methods_load = [["mean", 0.05]]
            orig = trained_model.transform(z, extr_methods_orig)
            new = loaded.transform(z, extr_methods_load)
            assert torch.allclose(orig, new, atol=1e-6)
        finally:
            os.unlink(path)

    def test_loaded_model_ps_not_empty(self, trained_model: Altx) -> None:
        """Loaded Ps dict contains entries for all RLK triplets."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trained_model.save(path)
            loaded = Altx.load(path)
            for rlk in loaded.RLK:
                assert rlk in loaded.Ps
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _multiply
# ---------------------------------------------------------------------------


class TestMultiply:
    """Tests for Altx._multiply."""

    def test_output_shape(self, trained_model: Altx) -> None:
        """Output shape is (nol_tilde, n_laws, m)."""
        torch.manual_seed(5)
        z = torch.randn(1, 20)
        rlk = trained_model.RLK[0]
        r, l, k = rlk
        M = trained_model._multiply(z, rlk)
        step = (r - 1) // (2 * l - 2)
        nol_tilde = (z.shape[1] - step * l + 1) // k
        n_laws = trained_model.Ps[rlk][1].shape[1]
        assert M.shape == (nol_tilde, n_laws, trained_model.m)

    def test_output_is_2d_when_m_equals_1(self, trained_model: Altx) -> None:
        """For a single-channel instance the third dimension of M is 1."""
        z = torch.randn(1, 20)
        rlk = trained_model.RLK[0]
        M = trained_model._multiply(z, rlk)
        assert M.shape[2] == 1


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


class TestTransform:
    """Tests for Altx.transform."""

    def test_output_is_1d_tensor(self, trained_model: Altx) -> None:
        """transform returns a 1-D feature vector."""
        torch.manual_seed(6)
        z = torch.randn(20)
        result = trained_model.transform(z, [["mean", 0.05]])
        assert result.ndim == 1

    def test_output_length(self, trained_model: Altx) -> None:
        """Feature length equals len(RLK) * noc * n_methods * m."""
        torch.manual_seed(6)
        z = torch.randn(20)
        extr_methods = [["mean", 0.05], ["var", 0.1]]
        result = trained_model.transform(z, extr_methods)
        n_methods = len(extr_methods)
        expected_len = (
            len(trained_model.RLK) * trained_model.noc * n_methods * trained_model.m
        )
        assert result.shape[0] == expected_len

    def test_numpy_input_accepted(self, trained_model: Altx) -> None:
        """NumPy array input is converted and processed correctly."""
        np.random.seed(7)
        z = np.random.randn(20).astype(np.float32)
        result = trained_model.transform(z, [["mean", 0.05]])
        assert isinstance(result, torch.Tensor)

    def test_1d_input_is_auto_unsqueezed(self, trained_model: Altx) -> None:
        """1-D input (single channel, no explicit channel dim) works."""
        z = torch.randn(20)
        assert z.ndim == 1
        result = trained_model.transform(z, [["mean", 0.05]])
        assert result.ndim == 1

    def test_channel_mismatch_raises(self, trained_model: Altx) -> None:
        """Instance with wrong channel count raises ValueError."""
        z = torch.randn(2, 20)  # 2 channels, but model trained on 1
        with pytest.raises(ValueError, match="first dimension"):
            trained_model.transform(z, [["mean", 0.05]])

    def test_mean_all_method(self, trained_model: Altx) -> None:
        """mean_all extraction method works end-to-end in transform."""
        z = torch.randn(20)
        result = trained_model.transform(z, [["mean_all", None]])
        assert result.ndim == 1

    def test_deterministic_output(self, trained_model: Altx) -> None:
        """Same instance produces identical features on repeated calls."""
        torch.manual_seed(8)
        z = torch.randn(20)
        r1 = trained_model.transform(z, [["mean", 0.05]])
        r2 = trained_model.transform(z, [["mean", 0.05]])
        assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# transform_set
# ---------------------------------------------------------------------------


class TestTransformSet:
    """Tests for Altx.transform_set."""

    def test_output_shape(self, trained_model: Altx) -> None:
        """Output is (n_instances, n_features)."""
        torch.manual_seed(9)
        test_set = torch.randn(3, 20)
        result = trained_model.transform_set(test_set, [["mean", 0.05]])
        n_features = len(trained_model.RLK) * trained_model.noc * 1 * trained_model.m
        assert result.shape == (3, n_features)

    def test_numpy_input_accepted(self, trained_model: Altx) -> None:
        """NumPy array test_set is accepted."""
        np.random.seed(10)
        test_set = np.random.randn(3, 20).astype(np.float32)
        result = trained_model.transform_set(test_set, [["mean", 0.05]])
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3

    def test_channel_mismatch_raises(self, trained_model: Altx) -> None:
        """Wrong channel count in test set raises ValueError."""
        test_set = torch.randn(3, 2, 20)  # 2 channels vs 1 in training
        with pytest.raises(ValueError, match="second dimension"):
            trained_model.transform_set(test_set, [["mean", 0.05]])

    def test_save_requires_test_classes(self, trained_model: Altx, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Passing save_file_name without test_classes raises TypeError."""
        test_set = torch.randn(3, 20)
        save_path = str(tmp_path / "out.csv")
        with pytest.raises(TypeError, match="classes"):
            trained_model.transform_set(
                test_set,
                [["mean", 0.05]],
                save_file_name=save_path,
                save_file_mode="New file",
            )

    def test_save_new_file_creates_csv(self, trained_model: Altx, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Saving with 'New file' mode creates a CSV with the expected structure."""
        torch.manual_seed(11)
        test_set = torch.randn(3, 20)
        test_classes = torch.tensor([0, 1, 0])
        save_path = str(tmp_path / "features.csv")
        trained_model.transform_set(
            test_set,
            [["mean", 0.05]],
            save_file_name=save_path,
            save_file_mode="New file",
            test_classes=test_classes,
        )
        assert os.path.exists(save_path)

    def test_save_new_file_has_class_column(
        self, trained_model: Altx, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """CSV file created by transform_set contains a 'Class' column."""
        import pandas as pd

        torch.manual_seed(12)
        test_set = torch.randn(3, 20)
        test_classes = torch.tensor([0, 1, 0])
        save_path = str(tmp_path / "features.csv")
        trained_model.transform_set(
            test_set,
            [["mean", 0.05]],
            save_file_name=save_path,
            save_file_mode="New file",
            test_classes=test_classes,
        )
        df = pd.read_csv(save_path)
        assert "Class" in df.columns
        assert len(df) == 3


# ---------------------------------------------------------------------------
# _generate_header
# ---------------------------------------------------------------------------


class TestGenerateHeader:
    """Tests for Altx._generate_header."""

    def test_last_column_is_class(self, trained_model: Altx) -> None:
        """The last element of the header is always 'Class'."""
        header = trained_model._generate_header([["mean", 0.05]])
        assert header[-1] == "Class"

    def test_header_length(self, trained_model: Altx) -> None:
        """Header length = n_RLK * noc * n_methods * m + 1 (for 'Class')."""
        extr_methods = [["mean", 0.05], ["var", 0.1]]
        header = trained_model._generate_header(extr_methods)
        n_expected = (
            len(trained_model.RLK) * trained_model.noc * 2 * trained_model.m + 1
        )
        assert len(header) == n_expected

    def test_header_contains_rlk_string(self, trained_model: Altx) -> None:
        """Each feature column name encodes the RLK triplet."""
        header = trained_model._generate_header([["mean", 0.05]])
        feature_headers = header[:-1]
        for col in feature_headers:
            assert "RLK" in col

    def test_header_contains_class_labels(self, trained_model: Altx) -> None:
        """Each feature column name encodes a class label."""
        header = trained_model._generate_header([["mean", 0.05]])
        feature_headers = header[:-1]
        for col in feature_headers:
            assert "C0" in col or "C1" in col

    def test_mean_all_omits_percentile(self, trained_model: Altx) -> None:
        """mean_all headers do not include a percentile suffix."""
        header = trained_model._generate_header([["mean_all", None]])
        feature_headers = header[:-1]
        for col in feature_headers:
            assert "(q" not in col


# ---------------------------------------------------------------------------
# _save_features
# ---------------------------------------------------------------------------


class TestSaveFeatures:
    """Tests for Altx._save_features."""

    @pytest.fixture
    def feature_data(self, trained_model: Altx) -> tuple:
        """Return (trained_model, features, test_classes) for save tests."""
        torch.manual_seed(13)
        test_set = torch.randn(3, 20)
        test_classes = torch.tensor([0, 1, 0])
        extr_methods = [["mean", 0.05]]
        features = trained_model.transform_set(test_set, extr_methods)
        return trained_model, features, test_classes, extr_methods

    def test_new_file_creates_csv(self, feature_data: tuple, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """'New file' mode creates a CSV at the specified path."""
        model, features, classes, methods = feature_data
        path = str(tmp_path / "new.csv")
        model._save_features(methods, features, classes, path, "New file")
        assert os.path.exists(path)

    def test_new_file_row_count(self, feature_data: tuple, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Saved CSV has one row per test instance."""
        import pandas as pd

        model, features, classes, methods = feature_data
        path = str(tmp_path / "rows.csv")
        model._save_features(methods, features, classes, path, "New file")
        df = pd.read_csv(path)
        assert len(df) == 3

    def test_append_instance_grows_row_count(
        self, feature_data: tuple, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """'Append instance' mode adds rows to an existing file.

        Notes
        -----
        The append writes a pandas index column, so the resulting CSV is
        not re-parseable with ``pd.read_csv``.  Line count is used instead.
        """
        model, features, classes, methods = feature_data
        path = str(tmp_path / "append_inst.csv")
        model._save_features(methods, features, classes, path, "New file")
        with open(path) as fh:
            initial_lines = sum(1 for _ in fh)
        model._save_features(methods, features, classes, path, "Append instance")
        with open(path) as fh:
            final_lines = sum(1 for _ in fh)
        # pandas df.to_csv(mode="a") writes a header row + 3 data rows
        assert final_lines == initial_lines + 4

    def test_append_feature_grows_column_count(
        self, feature_data: tuple, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """'Append feature' mode adds columns for the new extraction method."""
        import pandas as pd

        model, features, classes, _ = feature_data
        path = str(tmp_path / "append_feat.csv")
        # Write initial file with "mean" features
        model._save_features([["mean", 0.05]], features, classes, path, "New file")
        initial_df = pd.read_csv(path)
        initial_cols = len(initial_df.columns)
        # Append features from "var" method
        torch.manual_seed(14)
        test_set = torch.randn(3, 20)
        var_features = model.transform_set(test_set, [["var", 0.1]])
        model._save_features(
            [["var", 0.1]], var_features, classes, path, "Append feature"
        )
        final_df = pd.read_csv(path)
        # One extra feature column per (RLK * noc * m) combination was added
        extra = len(model.RLK) * model.noc * model.m
        assert len(final_df.columns) == initial_cols + extra

    def test_invalid_mode_raises(self, feature_data: tuple, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """An unrecognised save_file_mode raises ValueError."""
        model, features, classes, methods = feature_data
        path = str(tmp_path / "bad_mode.csv")
        # Create the file first so mode check is reached
        model._save_features(methods, features, classes, path, "New file")
        with pytest.raises(ValueError, match="save_file_mode"):
            model._save_features(methods, features, classes, path, "bad mode")

    def test_test_classes_length_mismatch_raises(
        self, feature_data: tuple, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        """Mismatched test_classes length raises ValueError."""
        model, features, _, methods = feature_data
        wrong_classes = torch.tensor([0, 1])  # 2 instead of 3
        path = str(tmp_path / "mismatch.csv")
        with pytest.raises(ValueError, match="does not match"):
            model._save_features(methods, features, wrong_classes, path, "New file")


# ---------------------------------------------------------------------------
# print_number_of_laws
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# multiply_only
# ---------------------------------------------------------------------------


class TestMultiplyOnly:
    """Tests for Altx.multiply_only."""

    def test_output_shape(self, trained_model: Altx) -> None:
        """Output shape is (nol_tilde, n_laws, m)."""
        torch.manual_seed(20)
        z = torch.randn(20)
        rlk = trained_model.RLK[0]
        r, l, k = rlk
        step = (r - 1) // (2 * l - 2)
        nol_tilde = (z.shape[0] - step * (l - 1) - 1) // k + 1
        n_laws = trained_model.Ps[rlk][1].shape[1]
        result = trained_model.multiply_only(z, rlk)
        assert result.shape == (nol_tilde, n_laws, trained_model.m)

    def test_output_dtype_is_float32(self, trained_model: Altx) -> None:
        """Output tensor always has float32 dtype."""
        z = torch.randn(20, dtype=torch.float64)
        result = trained_model.multiply_only(z, trained_model.RLK[0])
        assert result.dtype == torch.float32

    def test_numpy_input_accepted(self, trained_model: Altx) -> None:
        """NumPy array input is converted and produces a torch.Tensor output."""
        np.random.seed(21)
        z = np.random.randn(20).astype(np.float32)
        result = trained_model.multiply_only(z, trained_model.RLK[0])
        assert isinstance(result, torch.Tensor)

    def test_numpy_and_tensor_inputs_agree(self, trained_model: Altx) -> None:
        """NumPy and equivalent torch.Tensor inputs produce identical outputs."""
        np.random.seed(22)
        z_np = np.random.randn(20).astype(np.float32)
        z_t = torch.tensor(z_np)
        rlk = trained_model.RLK[0]
        assert torch.allclose(
            trained_model.multiply_only(z_np, rlk),
            trained_model.multiply_only(z_t, rlk),
        )

    def test_1d_input_auto_unsqueezed(self, trained_model: Altx) -> None:
        """1-D input is treated as a single-channel instance without error."""
        z = torch.randn(20)
        assert z.ndim == 1
        result = trained_model.multiply_only(z, trained_model.RLK[0])
        assert result.ndim == 3

    def test_normalize_false_differs_from_true(self, trained_model: Altx) -> None:
        """Disabling normalization produces a different result than the default."""
        torch.manual_seed(23)
        z = torch.randn(20)
        rlk = trained_model.RLK[0]
        result_norm = trained_model.multiply_only(z, rlk, normalize_data=True)
        result_raw = trained_model.multiply_only(z, rlk, normalize_data=False)
        assert not torch.allclose(result_norm, result_raw)

    def test_deterministic_output(self, trained_model: Altx) -> None:
        """Repeated calls with the same input return identical tensors."""
        torch.manual_seed(24)
        z = torch.randn(20)
        rlk = trained_model.RLK[0]
        assert torch.equal(
            trained_model.multiply_only(z, rlk),
            trained_model.multiply_only(z, rlk),
        )


# ---------------------------------------------------------------------------
# print_number_of_laws
# ---------------------------------------------------------------------------


class TestPrintNumberOfLaws:
    """Tests for Altx.print_number_of_laws."""

    def test_raises_before_training(self, model: Altx) -> None:
        """Calling before train() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="trained first"):
            model.print_number_of_laws()

    def test_runs_after_training(self, trained_model: Altx, capsys) -> None:  # type: ignore[no-untyped-def]
        """Calling after train() prints output without raising."""
        trained_model.print_number_of_laws()
        captured = capsys.readouterr()
        assert "Number of laws" in captured.out
