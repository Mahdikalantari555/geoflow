"""PyTorch integration utilities for geospatial batch processing."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


class GeoCollateUtils:
    """Custom collate functions for geospatial batch processing."""

    @staticmethod
    def geo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for GeoTensor batches.

        Handles stacking tensors with proper dimensions and preserving
        CRS and transform metadata.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Batched dictionary with stacked tensors and metadata.
        """
        inputs = []
        targets = []
        metadata_list = []

        for sample in batch:
            if "input" in sample:
                if hasattr(sample["input"], "data"):
                    inputs.append(sample["input"].data)
                elif isinstance(sample["input"], torch.Tensor):
                    inputs.append(sample["input"])
            if "target" in sample and sample["target"] is not None:
                if hasattr(sample["target"], "data"):
                    targets.append(sample["target"].data)
                elif isinstance(sample["target"], torch.Tensor):
                    targets.append(sample["target"])
                elif isinstance(sample["target"], int):
                    targets.append(torch.tensor(sample["target"]))
            metadata_list.append(sample.get("metadata", {}))

        result: Dict[str, Any] = {
            "metadata": metadata_list,
        }

        if inputs:
            result["inputs"] = torch.stack(inputs, dim=0) if inputs[0].dim() > 0 else torch.stack(inputs)

        if targets:
            if isinstance(targets[0], torch.Tensor):
                if targets[0].dim() > 0:
                    result["targets"] = torch.stack(targets, dim=0)
                else:
                    result["targets"] = torch.stack(targets)
            else:
                result["targets"] = torch.tensor(targets)

        return result

    @staticmethod
    def variable_size_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for variable-sized geospatial inputs.

        Pads all inputs to the maximum size in the batch.

        Args:
            batch: List of samples with potentially different sizes.

        Returns:
            Batched dictionary with padded tensors.
        """
        inputs = []
        targets = []
        metadata_list = []
        original_sizes = []

        for sample in batch:
            inp = sample.get("input")
            if hasattr(inp, "data"):
                tensor = inp.data
            elif isinstance(inp, torch.Tensor):
                tensor = inp
            else:
                continue

            inputs.append(tensor)
            original_sizes.append(tensor.shape[-2:])
            tgt = sample.get("target")
            if tgt is not None:
                if hasattr(tgt, "data"):
                    targets.append(tgt.data)
                elif isinstance(tgt, torch.Tensor):
                    targets.append(tgt)
            metadata_list.append(sample.get("metadata", {}))

        if not inputs:
            return {"inputs": torch.empty(0), "metadata": metadata_list}

        max_h = max(s.shape[-2] for s in inputs)
        max_w = max(s.shape[-1] for s in inputs)

        padded_inputs = []
        attention_masks = []

        for tensor in inputs:
            h, w = tensor.shape[-2:]
            pad_h = max_h - h
            pad_w = max_w - w
            if pad_h > 0 or pad_w > 0:
                padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
            else:
                padded = tensor
            padded_inputs.append(padded)

            mask = torch.ones(h, w, dtype=torch.bool)
            if pad_h > 0 or pad_w > 0:
                mask = F.pad(mask, (0, pad_w, 0, pad_h), value=False)
            attention_masks.append(mask)

        result: Dict[str, Any] = {
            "inputs": torch.stack(padded_inputs, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
            "metadata": metadata_list,
            "original_sizes": original_sizes,
        }

        if targets:
            padded_targets = []
            for tensor in targets:
                h, w = tensor.shape[-2:] if tensor.dim() >= 2 else (1, 1)
                if tensor.dim() >= 2:
                    pad_h = max_h - h
                    pad_w = max_w - w
                    if pad_h > 0 or pad_w > 0:
                        padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=-100)
                    else:
                        padded = tensor
                    padded_targets.append(padded)
            if padded_targets:
                result["targets"] = torch.stack(padded_targets, dim=0)

        return result