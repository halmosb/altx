"""Implementation of the `altx` algorithm."""

from __future__ import annotations

import gc
import os
import pickle
from typing import cast

import numpy as np
import pandas as pd
import torch

from .extract_methods import ExtractMethods, ExtrMethod, ExtrMethods


class Altx:
    """Implement the Adaptive Law-Based Transformation.

    Find the preserved quantities of the time series and use them to
    transform the test instances and prepare them for future
    classification and/or anomaly detection.

    Attributes
    ----------
    train_set : torch.Tensor
        The linear laws are based on this time series database.
    train_classes : torch.Tensor
        Contains the predefined class labels.
    train_length : torch.Tensor
        Length of train instances. Used when the length varies.
    noc : int
        The number of unique classes.
    class_labels : torch.Tensor
        The list of unique class labels.
    RLK : tuple
        Two-dimensional tuple for storing the used r-l-k triplets,
        where r is the length of the analyzed time window (always a
        multiple of 2*l-1), l is the dimension of the extracted laws,
        and k is the step of the time window.
    Ps : dict
        Contains the laws for r-l-k triplets, and a tensor that
        indicates which law belongs to the specific classes.
    tau : int
        Number of training instances. (Positive in each case.)
    m : int
        Number of channels. (Positive in each case.)
    device : torch.device
        Device where the operations will be carried out. When cuda,
        results may need to be moved to the cpu for further work.

    Methods
    -------
    _embed(instance_index, sensor_index, rlk, t=0)
        Embed the given time-window in a real, symmetric matrix.
    _get_law(S)
        Return the eigenvector for the smallest absolute eigenvalue.
    _get_P(rlk)
        Extract and store the laws for an embedding size in tensor P.
    _nol(r, k)
        Calculate the number of laws for a given r-k pair.
    train()
        Store tensors into a dictionary for each rlk triplet.
    save(save_file_name)
        Save the trained model with pickle into the given file.
    load(load_file_name)
        Load a previously trained and saved model.
    transform(z, extr_methods)
        Transform one instance into features with the given methods.
    transform_set(test_set, extr_method, save_file_name,
                  save_file_mode, test_classes)
        Transform a whole set by iterating the transform function.
    _save_features(extr_methods, features, test_classes,
                   save_file_name, save_file_mode)
        Save features to a CSV file.
    _generate_header(extr_methods)
        Generate the header columns for the feature save file.
    _multiply(z, rlk)
        Embed and multiply an instance with the P matrix for the rlk.
    _extract_features(M, extr_methods)
        Extract features from the result of the multiply function.
    plot(z, rlk, zoom)
        Transform one instance and plot the resulting matrix values.
    plot_anomalies(z)
        Not implemented yet. Will be used for anomaly detection.
    print_number_of_laws()
        Print the number of laws.

    Notes
    -----
    One instance contains m number of time series.

    """

    def __init__(
        self,
        train_set: torch.Tensor | np.ndarray,
        train_classes: torch.Tensor | np.ndarray,
        train_length: torch.Tensor | None = None,
        R: int | list[int] | None = None,
        L: list[int] | int = (5),
        K: int | list[int] = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the Altx model with training data and hyperparameters.

        Parameters
        ----------
        train_set : torch.Tensor or np.ndarray
            Training time series database used to extract linear laws.
        train_classes : torch.Tensor or np.ndarray
            Predefined class labels for each training instance.
        train_length : torch.Tensor, optional
            Length of each training instance. Used when lengths vary.
            If None, all instances are assumed to have the same length.
        R : int or list of int, optional
            Length of the analyzed time window. Must satisfy the form
            step*(2*L-2)+1. If None, defaults to 2*L-1 for each L.
        L : int or list of int, optional
            Dimension of the extracted laws. Default is (5).
        K : int or list of int, optional
            Step between consecutive time windows. Default is 1.
        device : torch.device or str, optional
            Device for computations. Default is CPU.

        Raises
        ------
        ValueError
            If train_set and train_classes have mismatched lengths, if
            train_length has a wrong shape, if R and L have different
            lengths, if R, L, or K contain non-positive values, if R
            is larger than the shortest training instance, or if R
            does not satisfy the form step*(2*L-2)+1.
        TypeError
            If K and L types are incompatible or R has an invalid type.

        Examples
        --------
        >>> import torch
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> train_data = torch.randn(6, 50)
        >>> train_classes = torch.tensor([0, 0, 0, 1, 1, 1])
        >>> model = Altx(train_data, train_classes, L=3, K=1)
        >>> model.noc
        2
        >>> model.m
        1
        >>> model.RLK
        ((5, 3, 1),)
        """
        if device is str:
            self.device = torch.device(device)
        else:
            self.device = device

        if type(train_set) is np.ndarray:
            train_set = torch.tensor(train_set)
        if type(train_classes) is np.ndarray:
            train_classes = torch.tensor(train_classes)
        if train_set.shape[0] != train_classes.shape[0]:
            raise ValueError(
                "Train classes and train set should have the same "
                "length along the first (instance) axis"
            )
        self.train_set: torch.Tensor = train_set.to(self.device)
        self.train_classes: torch.Tensor = train_classes.to(self.device)

        if len(self.train_set.shape) == 2:
            self.train_set = torch.unsqueeze(self.train_set, 1)

        if train_length is None:
            self.train_length = torch.full(
                size=tuple(self.train_classes.shape),
                fill_value=self.train_set.shape[2],
                dtype=torch.int,
            )
        elif len(train_length.shape) != 1:
            raise ValueError(
                "The train_length expected to have only 1 dimension, "
                f"not {len(train_length.shape)}"
            )
        elif train_length.shape[0] != train_classes.shape[0]:
            raise ValueError(
                "The train_length expected to have the same dimension "
                f"as train classes ({train_classes.shape}), "
                f"but got {train_length.shape}"
            )
        else:
            self.train_length = train_length

        # Number of classes
        self.class_labels = self.train_classes.unique()
        self.noc = self.class_labels.shape[0]

        self.tau, self.m, _ = self.train_set.shape

        if type(K) is int and type(L) is int:
            K = [K]
            L = [L]
        elif type(K) is int and type(L) is list:
            L = L
            K = [K] * len(L)
        elif type(K) is list and type(L) is int:
            L = [L] * len(K)
            K = K
        elif type(K) is list and type(L) is list and len(K) == len(L):
            L = L
            K = K
        else:
            raise TypeError(
                "Expected two ints, int and a list, or two lists with "
                f"the same size for K and L, but got {type(L)} and {type(K)}"
            )

        if R is None:
            R = [L[i] * 2 - 1 for i in range(len(L))]
        elif type(R) is int:
            R = [R] * len(L)
        elif type(R) is list:
            if len(R) == len(L):
                R = R
            else:
                raise ValueError("R and L should have the same length")
        else:
            raise TypeError(f"R should be None, int or list, but got {type(R)}")

        max_R = torch.min(self.train_length).item()

        for i in range(len(L)):
            if type(L[i]) is not int or type(K[i]) is not int:
                raise TypeError("The given `L' and `K' values must be integers.")
            if L[i] < 0 or K[i] < 0 or R[i] < 0:
                raise ValueError("The given `R', `L' and `K' values must be positive.")
            if R[i] > max_R:
                raise ValueError(
                    f"The given r ({R[i]}) and l ({L[i]}) are too large, "
                    f"the maximums are: {max_R} and {(max_R + 1) // 2}."
                )
            if (R[i] - 1) % (L[i] * 2 - 2) != 0:
                raise ValueError(
                    "Every R should have the form step*(2*L-2)+1, "
                    f"but got L: {L[i]}, R: {R[i]}"
                )

        self.RLK = tuple((R[i], L[i], K[i]) for i in range(len(L)))

        self.Ps: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _embed(
        self,
        instance_index: int,
        sensor_index: int,
        rlk: tuple[int, int, int],
        t: int = 0,
    ) -> torch.Tensor:
        """Embed the given time-window in a real, symmetric matrix.

        Parameters
        ----------
        instance_index : int
            The index of the instance to embed.
        sensor_index : int
            The index of the time series in the instance to embed.
        rlk : tuple of int
            The (r, l, k) triplet used for the embedding.
            The k component is not used here.
        t : int, optional
            The start index in the time series for the embedding.
            Default is 0.

        Returns
        -------
        torch.Tensor
            The embedded matrix.
        """
        r, l, _ = rlk
        # Step between datapoints used in the embedding
        step = (r - 1) // (2 * l - 2)
        data = self.train_set[instance_index, sensor_index, t : t + r : step]
        S = torch.stack([data[i : i + l] for i in range(l)])
        return S

    def _get_law(self, S: torch.Tensor) -> torch.Tensor:
        """Return the eigenvector for the smallest absolute eigenvalue.

        Compute the eigenvalue decomposition of the given real,
        symmetric matrix and return the eigenvector corresponding to
        the smallest absolute-valued eigenvalue.

        Parameters
        ----------
        S : torch.Tensor
            The input matrix.

        Returns
        -------
        torch.Tensor
            The eigenvector.

        Raises
        ------
        torch._C._LinAlgError
            If the eigenvalue decomposition does not converge.
        """
        L, V = torch.linalg.eigh(S)
        L = L**2
        idx = L.argmin()
        return V[:, idx]

    def _get_P(self, rlk: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the laws for a given (r, l, k) triplet.

        Parameters
        ----------
        rlk : tuple of int
            The (r, l, k) triplet.

        Returns
        -------
        P_classes : torch.Tensor
            The class label corresponding to each column in P.
        P : torch.Tensor
            The generated tensor of laws.
        """
        r, l, k = rlk
        # The nol is an upper bound of the number of laws
        nol = self._nol(r, k)
        # Initializing the P tensor, and P_classes tensor
        P = torch.zeros((l, nol * self.tau, self.m), device=self.device)
        P_classes = torch.full((P.shape[1],), torch.nan, device=self.device)
        for tau_i in range(self.tau):  # For each instance
            for m_i in range(self.m):
                for t in range(
                    0, self.train_length[tau_i] - r + 1, k
                ):  # For each starting time of embedding
                    S = self._embed(tau_i, m_i, rlk, t)  # We embed the matrix
                    try:
                        law = self._get_law(S)
                        P[:, tau_i * nol + t // k, m_i] = law
                        P_classes[tau_i * nol + t // k] = self.train_classes[
                            tau_i
                        ]  # And store the resulting law
                    except torch._C._LinAlgError as e:
                        # If the eigenvalue-decomposition does not converge
                        print(e)
        # TODO: Remove laws here
        P = P[:, torch.logical_not(P_classes.isnan()), :]
        P_classes = P_classes[torch.logical_not(P_classes.isnan())]
        return (P_classes, P)

    def _nol(self, r: int, k: int) -> int:
        """Calculate the number of laws for an instance with maximal length.

        Parameters
        ----------
        r : int
            The time window length.
        k : int
            The step between consecutive windows.

        Returns
        -------
        int
            The number of laws from one instance.
        """
        # The maximal length of a time series in the train_set
        t = self.train_set.shape[2]
        res: int = (t - r + 1) // k + 1
        return res

    def save(self, save_file_name: str) -> None:
        """Save the trained model to a file.

        Parameters
        ----------
        save_file_name : str
            Path of the save file.

        Examples
        --------
        >>> import torch, tempfile, os
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> model = Altx(
        ...     torch.randn(6, 50),
        ...     torch.tensor([0, 0, 0, 1, 1, 1]),
        ...     L=3, K=1,
        ... )
        >>> model.train()
        >>> path = tempfile.mktemp(suffix=".pkl")
        >>> model.save(path)
        >>> os.path.exists(path)
        True
        >>> os.unlink(path)
        """
        model = {
            "RLK": self.RLK,
            "m": self.m,
            "Ps": self.Ps,
            "noc": self.noc,
            "class_labels": self.class_labels,
        }
        with open(save_file_name, "wb") as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(load_file_name: str) -> Altx:
        """Load a previously trained and saved model.

        Parameters
        ----------
        load_file_name : str
            Path of the save file.

        Returns
        -------
        Altx
            The trained model instance.

        Notes
        -----
        The save file does not contain the training data, so further
        training is not possible after loading.

        Examples
        --------
        >>> import torch, tempfile, os
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> model = Altx(
        ...     torch.randn(6, 50),
        ...     torch.tensor([0, 0, 0, 1, 1, 1]),
        ...     L=3, K=1,
        ... )
        >>> model.train()
        >>> path = tempfile.mktemp(suffix=".pkl")
        >>> model.save(path)
        >>> loaded = Altx.load(path)
        >>> loaded.device = model.device
        >>> loaded.noc
        2
        >>> loaded.RLK
        ((5, 3, 1),)
        >>> os.unlink(path)
        """
        with open(load_file_name, "rb") as file:
            model = pickle.load(file)
            sbst = Altx.__new__(Altx)
            sbst.m = model["m"]
            sbst.RLK = model["RLK"]
            sbst.Ps = model["Ps"]
            sbst.noc = model["noc"]
            sbst.class_labels = model["class_labels"]
            return sbst

    def train(self, cleanup: bool = False) -> None:
        """Train the model by extracting laws from the training data.

        Extract and store the patterns (laws) for each (r, l, k)
        triplet.

        Parameters
        ----------
        cleanup : bool, optional
            Whether to delete the training data after training to free
            memory. Default is False.

        Raises
        ------
        RuntimeError
            If training is attempted without training data.

        Examples
        --------
        >>> import torch
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> model = Altx(
        ...     torch.randn(6, 50),
        ...     torch.tensor([0, 0, 0, 1, 1, 1]),
        ...     L=3, K=1,
        ... )
        >>> model.train()
        >>> (5, 3, 1) in model.Ps
        True
        >>> _, P = model.Ps[(5, 3, 1)]
        >>> P.shape
        torch.Size([3, 276, 1])
        """
        if self.train_set is None:
            raise RuntimeError(
                "The model cannot be trained after loading, "
                "as the training data is not saved."
            )
        for rlk in self.RLK:
            self.Ps[rlk] = self._get_P(rlk)
        if cleanup:
            del self.train_set
            del self.train_classes
            del self.tau
            del self.train_length
            gc.collect()
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

    def _multiply(self, z: torch.Tensor, rlk: tuple[int, int, int]) -> torch.Tensor:
        """Multiply an instance with the generated laws.

        Parameters
        ----------
        z : torch.Tensor
            An instance of time series.
        rlk : tuple of int
            The (r, l, k) triplet.

        Returns
        -------
        torch.Tensor
            A tensor of the multiplication results.
        """
        r, l, k = rlk
        # Step between datapoints used in the embedding
        step = (r - 1) // (2 * l - 2)
        # The length of the embedding of z along the first dimension
        nol_tilde = (z.shape[1] - step * l + 1) // k
        data = torch.stack(
            [z[:, i * k : i * k + step * l : step].T for i in range(nol_tilde)]
        )
        return torch.einsum("kij, ilj -> klj", data, self.Ps[rlk][1])

    def _extract_features(
        self,
        M: torch.Tensor,
        extr_methods: ExtrMethods,
    ) -> torch.Tensor:
        """Extract features from _multiply results using all given methods.

        Parameters
        ----------
        M : torch.Tensor
            The result of the multiplication for one class.
        extr_methods : list of (list[str] or list[str, float])
            Each element is either a one-element list ``[method]`` or a
            two-element list ``[method, percentile]``. If the percentile
            is omitted, 0.05 is used by default.

        Returns
        -------
        torch.Tensor
            The calculated features in a two-dimensional tensor.

        Raises
        ------
        ValueError
            If the given method is not implemented in ExtractMethods.
        """
        results = ExtractMethods.extract(M, extr_methods, self.device)
        return results

    def transform(
        self,
        z: torch.Tensor | np.ndarray,
        extr_methods: ExtrMethods,
    ) -> torch.Tensor:
        """Transform one instance into features using the given methods.

        Parameters
        ----------
        z : torch.Tensor or np.ndarray
            The input time series instance.
        extr_methods : list of (list[str] or list[str, float])
            Each element is either a one-element list ``[method]`` or a
            two-element list ``[method, percentile]``. If the percentile
            is omitted, 0.05 is used by default.

        Returns
        -------
        torch.Tensor
            A one-dimensional tensor of the calculated features.

        Raises
        ------
        ValueError
            If the given extraction method is not implemented.

        Examples
        --------
        >>> import torch
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> model = Altx(
        ...     torch.randn(6, 50),
        ...     torch.tensor([0, 0, 0, 1, 1, 1]),
        ...     L=3, K=1,
        ... )
        >>> model.train()
        >>> _ = torch.manual_seed(0)
        >>> z = torch.randn(50)
        >>> features = model.transform(z, [["mean", 0.05]])
        >>> features.shape
        torch.Size([2])
        >>> features = model.transform(z, [["mean", 0.05], ["var", 0.1]])
        >>> features.shape
        torch.Size([4])
        """
        for i in range(len(extr_methods)):
            if extr_methods[i] == ["mean_all"]:
                extr_methods[i] = cast(ExtrMethod, extr_methods[i] + [None])
            elif len(extr_methods[i]) == 1:
                extr_methods[i] = cast(ExtrMethod, extr_methods[i] + [0.05])
        if type(z) is np.ndarray:
            z = torch.tensor(z)
        # Adding second dimension if it is single variate
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, 0)
        if z.shape[0] != self.m:
            raise ValueError(
                "The given set has not the same length along the "
                f"first dimension (got {z.shape[0]}) "
                f"as the training data ({self.m})."
            )
        # The number of extraction methods
        n = len(extr_methods)
        # Allocating tensor for features
        feature = torch.zeros((len(self.RLK), self.noc, n, self.m), device=self.device)
        # Extracting features for all (r, l, k)
        for i in range(len(self.RLK)):
            rlk = self.RLK[i]
            M = self._multiply(z, rlk)
            P_classes = self.Ps[rlk][0]
            # Separating the classes
            partitions = [M[:, P_classes == cls, :] for cls in self.class_labels]
            for j in range(self.noc):
                feature[i, j, :, :] = self._extract_features(
                    partitions[j], extr_methods
                )
        # Flattening the result
        return feature.flatten()

    def transform_set(
        self,
        test_set: torch.Tensor | np.ndarray,
        extr_methods: ExtrMethods,
        test_length: torch.Tensor | None = None,
        save_file_name: str | None = None,
        save_file_mode: str | None = None,
        test_classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform a whole set of instances by iterating transform.

        Save the features in CSV format if save parameters are given.

        Parameters
        ----------
        test_set : torch.Tensor or np.ndarray
            The set of instances to transform (same dimensions as
            train_set).
        extr_methods : list of (list[str] or list[str, float])
            Each element is either a one-element list ``[method]`` or a
            two-element list ``[method, percentile]``. If the percentile
            is omitted, 0.05 is used by default.
        test_length : torch.Tensor, optional
            The useful length of each instance in the set. Default is
            None, which uses the training set length.
        save_file_name : str, optional
            The path for the save file. Default is None (no saving).
        save_file_mode : str, optional
            Saving mode: "New file", "Append feature", or
            "Append instance". Ignored if save_file_name is None or
            the file does not exist.
        test_classes : torch.Tensor, optional
            The class labels for the transformed set. Required when
            saving.

        Returns
        -------
        torch.Tensor
            The results in a two-dimensional tensor.

        Raises
        ------
        ValueError
            If the given extraction method is not implemented.
        TypeError
            If save_file_name is given but test_classes is None.

        Examples
        --------
        >>> import torch
        >>> from altx import ALT as Altx
        >>> _ = torch.manual_seed(0)
        >>> model = Altx(
        ...     torch.randn(6, 50),
        ...     torch.tensor([0, 0, 0, 1, 1, 1]),
        ...     L=3, K=1,
        ... )
        >>> model.train()
        >>> _ = torch.manual_seed(0)
        >>> test_set = torch.randn(3, 50)
        >>> features = model.transform_set(test_set, [["mean", 0.05]])
        >>> features.shape
        torch.Size([3, 2])
        >>> features
        tensor([[0.0077, 0.0113],
                [0.0122, 0.0163],
                [0.0077, 0.0104]])
        """
        for i in range(len(extr_methods)):
            if extr_methods[i] in [["mean_all"]]:
                extr_methods[i] = cast(ExtrMethod, extr_methods[i] + [None])
            elif len(extr_methods[i]) == 1:
                extr_methods[i] = cast(ExtrMethod, extr_methods[i] + [0.05])

        if test_length is None:
            test_length = torch.full((test_set.shape[0],), self.train_set.shape[2])
        if type(test_set) is np.ndarray:
            test_set = torch.tensor(test_set, dtype=torch.float32)
        if type(test_classes) is np.ndarray:
            test_classes = torch.tensor(test_classes)
        # Calculating features
        if len(test_set.shape) == 2:
            test_set = torch.unsqueeze(test_set, 1)
        if test_set.shape[1] != self.m:
            raise ValueError(
                "The given set has not the same length along the "
                f"second dimension (got {test_set.shape[1]}) "
                f"as the training data ({self.m})."
            )
        test_set = test_set.to(self.device)
        # Iterating transform
        features = torch.stack(
            [
                self.transform(
                    test_set[i, :, : test_length[i]], extr_methods=extr_methods
                )
                for i in range(test_set.shape[0])
            ]
        )
        # Saving
        if save_file_name:
            if test_classes is not None:
                self._save_features(
                    extr_methods,
                    features,
                    test_classes,
                    save_file_name,
                    save_file_mode,
                )
            else:
                raise TypeError(
                    "When saving features you must provide the "
                    "classes of test instances."
                )
        return features

    def _save_features(
        self,
        extr_methods: ExtrMethods,
        features: torch.Tensor,
        test_classes: torch.Tensor,
        save_file_name: str,
        save_file_mode: str | None,
    ) -> None:
        """Save generated features to a CSV file.

        Parameters
        ----------
        extr_methods : list of (list[str] or list[str, float])
            The used extraction methods.
        features : torch.Tensor
            The features generated by transform_set.
        test_classes : torch.Tensor
            The class labels of the transformed instances.
        save_file_name : str
            The path of the save file.
        save_file_mode : str or None
            Can be "New file", "Append feature", or "Append instance".
            Ignored if save_file_name does not exist.

        Raises
        ------
        ValueError
            If test_classes length does not match the number of
            test instances, or if save_file_mode is invalid.

        Notes
        -----
        Creates a new file or edits an existing one depending on
        save_file_mode.
        """
        if features.shape[0] != test_classes.shape[0]:
            raise ValueError(
                f"test_classes length ({test_classes.shape[0]}) does not "
                f"match with number of test instances ({features.shape[0]})."
            )
        features_np = features.cpu().numpy()  # convert to Numpy array
        test_classes_np = test_classes.cpu().numpy()
        # New file, or if the file doesn't exist.
        if save_file_mode == "New file" or not os.path.exists(save_file_name):
            df = pd.DataFrame(features_np)  # convert to a dataframe
            df["Class"] = test_classes_np
            # n = len(list(extr_methods.keys()))
            feat_list = self._generate_header(extr_methods)
            df.to_csv(save_file_name, index=False, header=feat_list)
        # Append instance
        elif save_file_mode == "Append instance":
            df = pd.DataFrame(features_np)  # convert to a dataframe
            df["Class"] = test_classes_np
            df.to_csv(save_file_name, mode="a")  # save to file
        # Apppend feature
        elif save_file_mode == "Append feature":
            old_df = pd.read_csv(save_file_name, header=0)
            old_features = old_df.values[:, :-1]
            new_features = np.concatenate((old_features, features_np), axis=1)
            df = pd.DataFrame(new_features)  # convert to a dataframe
            # n = len(list(extr_methods.keys()))
            feat_list = list(old_df)[:-1] + self._generate_header(extr_methods)
            df["Class"] = test_classes_np
            df.to_csv(save_file_name, index=False, header=feat_list)
        # Default
        else:
            raise ValueError(
                "save_file_mode should be either 'New file', "
                "'Append instance' or 'Append feature' but got "
                f"{save_file_mode}"
            )

    def _generate_header(self, extr_methods: ExtrMethods) -> list[str]:
        """Generate the header for the feature save file.

        Parameters
        ----------
        extr_methods : list of (list[str] or list[str, float])
            The used extraction methods.

        Returns
        -------
        list of str
            The list of column names for the pandas DataFrame.
        """
        sep = "|"
        return [
            f"RLK{self.RLK[l_i]}{sep}C{c_i}{sep}F{method}"
            + ("" if method in ["mean_all"] else f"(q{q})")
            + f"{sep}m{m_i}"
            for l_i in range(len(self.RLK))
            for c_i in self.class_labels
            for method, q in extr_methods
            for m_i in range(self.m)
        ] + ["Class"]

    def print_number_of_laws(self) -> None:
        """Print the number of laws for each class and (r, l, k) triplet.

        Raises
        ------
        RuntimeError
            If called before training.

        Notes
        -----
        Only usable after training.
        """
        # Print the number of laws for each class and each l-k pair
        if self.RLK[0] not in self.Ps:
            raise RuntimeError("The model should be trained first.")
        for rlk in self.RLK:
            r, l, k = rlk
            print(f"Number of laws for r = {r}, l = {l}, k = {k}:")
            print(self.Ps[rlk][0].unique(return_counts=True))
