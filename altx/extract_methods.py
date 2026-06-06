"""Statistical feature extraction methods for the `altx` algorithm."""

from __future__ import annotations

from typing import cast

import torch

type ExtrMethod = list[str] | list[str | float]
type ExtrMethods = list[ExtrMethod]


class ExtractMethods:
    """Provide statistical feature extraction methods for Altx.

    Implement the most common statistical methods used for feature
    extraction in the Adaptive Law-Based Transformation.

    Methods
    -------
    extract(F, extr_methods, device)
        Extract features from F with the given extraction methods.
    nth_moment(percentiles, n)
        Calculate the n-th moment of percentiles along dimension 1.
    excess_kurtosis(percentiles)
        Calculate the excess kurtosis along dimension 1.
    """

    @staticmethod
    def excess_kurtosis(percentiles: torch.Tensor) -> torch.Tensor:
        """Calculate the excess kurtosis of percentiles along dimension 1.

        Parameters
        ----------
        percentiles : torch.Tensor
            Input tensor of pre-computed percentiles.

        Returns
        -------
        torch.Tensor
            The excess kurtosis of the computed percentiles.
        """
        mean = torch.mean(percentiles, dim=1)
        deviations = percentiles - mean
        fourth_moment = torch.mean(deviations**4, dim=1)
        variance = torch.mean(deviations**2, dim=1)
        kurt = fourth_moment / (variance**2)
        excess_kurtosis = kurt - 3

        # Check for NaN values
        if torch.isnan(excess_kurtosis).any():
            print("Nan values found in the computed excess kurtosis.")
            # raise ValueError(
            #     "NaN values found in the computed excess kurtosis."
            # )
            return torch.zeros_like(excess_kurtosis)
        return excess_kurtosis

    @staticmethod
    def nth_moment(percentiles: torch.Tensor, n: int = 4) -> torch.Tensor:
        """Calculate the n-th moment of percentiles along dimension 1.

        Parameters
        ----------
        percentiles : torch.Tensor
            Input tensor of pre-computed percentiles.
        n : int, optional
            The order of the moment to compute. Default is 4.

        Returns
        -------
        torch.Tensor
            The n-th moment of the computed percentiles.
        """
        mean = torch.mean(percentiles, dim=1)
        deviations = percentiles - mean
        nth_moment = torch.mean(deviations**n, dim=1)

        # Check for NaN values
        if torch.isnan(nth_moment).any():
            print(f"NaN values found in the computed moment. (n={n})")
            return torch.zeros_like(nth_moment)
        return nth_moment

    @staticmethod
    def extract(
        F: torch.Tensor,
        extr_methods: ExtrMethods,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Extract features from F using the given extraction methods.

        Parameters
        ----------
        F : torch.Tensor
            Input tensor.
        extr_methods : list of (list[str] or list[str, float])
            Each element is either a one-element list ``[method]`` or a
            two-element list ``[method, percentile]``.
        device : torch.device or str, optional
            The device to calculate on. Default is CPU.

        Returns
        -------
        torch.Tensor
            The tensor of the collected features.

        Raises
        ------
        ValueError
            If the given extraction method is not implemented.

        Notes
        -----
        The return tensor has the shape (n, m), where n is the number
        of used extraction methods and m is the size of the input
        tensor along the third dimension.
        """
        if type(device) is str:
            device = torch.device(device)
        F = F**2
        qs = [cast(float, s[1]) for s in extr_methods if s[1] is not None]
        if qs != []:
            q = torch.Tensor(qs).unique().to(device)
            percentiles = torch.quantile(F, q, dim=1)
        results = []
        for s in extr_methods:
            method = cast(str, s[0])
            perc = cast(float | None, s[1])
            if method == "mean":
                results.append(torch.mean(percentiles[q == perc], dim=1))
            elif method == "var":
                var_values = torch.var(percentiles[q == perc], dim=1)
                if var_values.isnan().any():
                    var_values = torch.zeros_like(var_values)
                    print(
                        "Error: NaN values were found in the result "
                        "of the var calculation!"
                    )
                results.append(var_values)
            elif method == "excess_kurtosis":
                results.append(ExtractMethods.excess_kurtosis(percentiles[q == perc]))
            elif method[-7:] == "_moment":
                results.append(
                    ExtractMethods.nth_moment(
                        percentiles[q == perc], n=int(method[:-9])
                    )
                )
            elif method == "mean_all":
                results.append(torch.mean(F, dim=(0, 1)).unsqueeze(0))
            else:
                raise ValueError(f"The method {method} is not implemented")
        # print(list(r.shape for r in results))
        return torch.cat(results)
