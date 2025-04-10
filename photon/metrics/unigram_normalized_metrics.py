"""Allows comparing between models with different vocabularies."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class PureUnigramCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.

    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        unigram_probabilities: Tensor,
        ignore_index: int = -100,
        *,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize the metric.

        Parameters
        ----------
        unigram_probabilities : torch.Tensor
            The unigram probabilities for the vocabulary.
        ignore_index : int, optional
            The class index to ignore, by default -100.
        dist_sync_on_step : bool, optional
            Synchronize metric state across processes at each forward() before returning
            the value at the step, by default False.

        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.unigram_probabilities = unigram_probabilities
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:  # noqa: ARG002
        """Update the internal state with results from a new batch.

        Args:
            output (Mapping): The output from the model, which must contain
                either the Tensor or a Mapping type
                    that contains the loss or model logits.
            target (~torch.Tensor): A Tensor of ground-truth values to compare against.

        """
        target = target.view(-1)

        self.unigram_probabilities = self.unigram_probabilities.to(target.device)

        probabilities = self.unigram_probabilities[target]
        unigram_cross_entropy = -torch.log(probabilities)

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items

        # Mask out the ignored indices
        mask = (target != self.ignore_index).float()
        unigram_cross_entropy = (unigram_cross_entropy * mask).sum()

        # accumulate loss over all batches
        self.sum_loss += unigram_cross_entropy

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns
        -------
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.

        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items


class PureUnigramPerplexity(PureUnigramCrossEntropy):
    """Implements unigram-normalized perplexity."""

    def compute(self) -> Tensor:
        """Return torch.exp() of the UnigramNormalizedLanguageCrossEntropy.

        Returns
        -------
        Tensor
            The perplexity of the model

        """
        avg_loss = super().compute()
        return torch.exp(avg_loss)


class UnigramNormalizedLanguageCrossEntropy(Metric):
    """Torchmetric that computes cross entropy on language modeling outputs.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.

    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        unigram_probabilities: Tensor,
        ignore_index: int = -100,
        *,
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize the metric.

        Parameters
        ----------
        unigram_probabilities : torch.Tensor
            The unigram probabilities for the vocabulary.
        ignore_index : int, optional
            The class index to ignore, by default -100.
        dist_sync_on_step : bool, optional
            Synchronize metric state across processes at each forward() before returning
            the value at the step, by default False.

        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.unigram_probabilities = unigram_probabilities
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="sum",
        )
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        """Update the internal state with results from a new batch.

        Parameters
        ----------
        output : Mapping | Tensor
            The output from the model, which must contain either the Tensor or a Mapping
            type that contains the loss or model logits.
        target : torch.Tensor
            A Tensor of ground-truth values to compare against.

        Raises
        ------
        TypeError
            If the output is not a Mapping or

        """
        if isinstance(output, Mapping):
            logits = output["logits"]
        elif isinstance(output, Tensor):
            logits = output
        else:
            msg = f"Type {type(output)} for the output is unsupported."
            raise TypeError(msg)

        target = target.view(-1)
        logits = logits.view(target.shape[0], -1)

        losses = self.loss_fn(logits, target)

        self.unigram_probabilities = self.unigram_probabilities.to(target.device)

        probabilities = self.unigram_probabilities[target]

        unigram_cross_entropy = -torch.log(probabilities)

        total_items = (target != self.ignore_index).sum()
        self.total_items += total_items

        # Mask out the ignored indices
        mask = (target != self.ignore_index).float()
        unigram_cross_entropy = (unigram_cross_entropy * mask).sum()

        # accumulate loss over all batches
        self.sum_loss += losses - unigram_cross_entropy

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns
        -------
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.

        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items


class UnigramNormalizedLanguagePerplexity(UnigramNormalizedLanguageCrossEntropy):
    """Implements unigram-normalized perplexity."""

    def compute(self) -> Tensor:
        """Return torch.exp() of the UnigramNormalizedLanguageCrossEntropy.

        Returns
        -------
        Tensor
            The perplexity of the model

        """
        avg_loss = super().compute()
        return torch.exp(avg_loss)


def create_wrapped_subclass(base_class: type, **kwargs: Any) -> type:  # noqa: ANN401
    """Create a subclass of a given class with additional kwargs.

    Parameters
    ----------
    base_class : type
        The base class to inherit from.
    **kwargs : dict
        Additional keyword arguments to pass to the subclass constructor.

    Returns
    -------
    type
        A subclass of `base_class` with the additional keyword arguments.

    """

    class SubclassWithKwargs(base_class):
        def __init__(self, **init_kwargs: Any) -> None:  # noqa: ANN401
            combined_kwargs = {**kwargs, **init_kwargs}
            super().__init__(**combined_kwargs)

    SubclassWithKwargs.__name__ = base_class.__name__
    return SubclassWithKwargs


UNIGRAM_METRIC_NAMES_AND_CLASSES = MappingProxyType({
    "UnigramNormalizedLanguageCrossEntropy": UnigramNormalizedLanguageCrossEntropy,
    "UnigramNormalizedLanguagePerplexity": UnigramNormalizedLanguagePerplexity,
    "PureUnigramCrossEntropy": PureUnigramCrossEntropy,
    "PureUnigramPerplexity": PureUnigramPerplexity,
})
