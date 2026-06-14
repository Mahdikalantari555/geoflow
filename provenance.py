"""Provenance tracking system for GeoTensor (§14 of GeoTensor Spec).

Provides :class:`ProvenanceGraph` based on W3C PROV-O, with export,
reproduction, and freeze functionality.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .core import GeoTensor, ImmutableGeoTensorError, to_geotensor


# ====================================================================
# Provenance Node
# ====================================================================


class ProvenanceNode:
    """A single operation node in the provenance graph.

    Attributes:
        node_id: Unique identifier (UUID).
        operation: Short name of the operation (e.g. ``'reproject'``).
        parameters: Dict of keyword arguments supplied to the operation.
        inputs: List of input node IDs.
        output: Optional output node ID.
        timestamp: When the operation was applied.
        software_versions: Dict of software versions used.
    """

    def __init__(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        inputs: Optional[List[str]] = None,
        output: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        software_versions: Optional[Dict[str, str]] = None,
        node_id: Optional[str] = None,
    ) -> None:
        self.node_id = node_id or str(uuid.uuid4())
        self.operation = operation
        self.parameters = parameters or {}
        self.inputs = inputs or []
        self.output = output
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.software_versions = software_versions or {
            'geotensor': '2.0.0',
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            'id': self.node_id,
            'operation': self.operation,
            'parameters': self.parameters,
            'inputs': self.inputs,
            'output': self.output,
            'timestamp': self.timestamp.isoformat(),
            'software_versions': self.software_versions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceNode':
        """Create from a dictionary."""
        return cls(
            operation=data['operation'],
            parameters=data.get('parameters', {}),
            inputs=data.get('inputs', []),
            output=data.get('output'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            software_versions=data.get('software_versions', {}),
            node_id=data.get('id'),
        )

    def __repr__(self) -> str:
        return (
            f'<ProvenanceNode operation={self.operation!r} '
            f'id={self.node_id[:8]}>'
        )


# ====================================================================
# ProvenanceGraph
# ====================================================================


class ProvenanceGraph:
    """W3C PROV-O inspired provenance graph.

    Stores an ordered sequence of :class:`ProvenanceNode` instances,
    each representing a transformation step.

    Attributes:
        nodes: Ordered list of provenance nodes.
        _current_output: The most recent output node ID.
        frozen: If ``True``, the associated GeoTensor is read-only.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._order: List[str] = []
        self._current_output: Optional[str] = None
        self.frozen: bool = False

    @property
    def nodes(self) -> List[ProvenanceNode]:
        """Ordered list of all provenance nodes."""
        return [self._nodes[nid] for nid in self._order]

    @property
    def graph(self) -> List[Dict[str, Any]]:
        """Serialized list of all operation records."""
        return [n.to_dict() for n in self.nodes]

    def record(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        inputs: Optional[List[str]] = None,
    ) -> str:
        """Record an operation in the provenance graph.

        Args:
            operation: Operation name.
            parameters: Operation parameters.
            inputs: Input node IDs.

        Returns:
            The new node's ID.
        """
        node = ProvenanceNode(
            operation=operation,
            parameters=parameters,
            inputs=inputs or ([self._current_output] if self._current_output else []),
        )
        self._nodes[node.node_id] = node
        self._order.append(node.node_id)
        self._current_output = node.node_id
        return node.node_id

    def check_not_frozen(self) -> None:
        """Raise :class:`ImmutableGeoTensorError` if frozen."""
        if self.frozen:
            raise ImmutableGeoTensorError(
                "This GeoTensor is frozen (read-only). "
                "Unfreeze or create a copy to modify."
            )

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f'<ProvenanceGraph nodes={len(self._nodes)} frozen={self.frozen}>'


# ====================================================================
# Provenance operations — called from accessor
# ====================================================================


def get_provenance(obj: Any) -> ProvenanceGraph:
    """Retrieve or create a :class:`ProvenanceGraph` from attrs.

    Args:
        obj: DataArray / GeoTensor (with ``attrs['provenance']``).

    Returns:
        The :class:`ProvenanceGraph` instance.
    """
    prov = obj.attrs.get('provenance')
    if isinstance(prov, ProvenanceGraph):
        return prov

    new_prov = ProvenanceGraph()
    obj.attrs['provenance'] = new_prov
    return new_prov


def record_operation(
    obj: Any,
    operation: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an operation in the provenance graph.

    Args:
        obj: DataArray / GeoTensor.
        operation: Operation name.
        parameters: Operation parameters.
    """
    prov = get_provenance(obj)
    prov.check_not_frozen()
    prov.record(operation, parameters=parameters)


def export_provenance(
    obj: Any,
    format: str = 'json',
) -> str:
    """Export provenance graph in the specified format.

    Args:
        obj: DataArray / GeoTensor.
        format: Output format (``'json'``, ``'json-ld'``, ``'dot'``).

    Returns:
        Provenance string representation.

    Raises:
        ValueError: If the format is unsupported.
    """
    prov = get_provenance(obj)
    graph_data = prov.graph

    if format == 'json':
        return json.dumps(graph_data, indent=2, default=str)
    elif format == 'json-ld':
        ld = {
            '@context': {
                'prov': 'http://www.w3.org/ns/prov#',
                'geotensor': 'https://geotensor.dev/ns/',
            },
            '@graph': graph_data,
        }
        return json.dumps(ld, indent=2, default=str)
    elif format == 'dot':
        lines = ['digraph Provenance {']
        for node in prov.nodes:
            label = f"{node.operation}\\n{node.node_id[:8]}"
            lines.append(f'  "{node.node_id}" [label="{label}"];')
            for inp in node.inputs:
                lines.append(f'  "{inp}" -> "{node.node_id}";')
        lines.append('}')
        return '\n'.join(lines)
    else:
        raise ValueError(f"Unsupported format: '{format}'. Use 'json', 'json-ld', or 'dot'.")


def reproduce(
    provenance_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    target_storage: Optional[str] = None,
) -> GeoTensor:
    """Reproduce a GeoTensor from a provenance record.

    Replays the operations stored in the provenance graph to reconstruct
    the GeoTensor. The first node must be a ``'source'`` or ``'open'``
    operation with a ``source`` or ``path`` parameter pointing to
    accessible data.

    Args:
        provenance_data: Path to JSON file, dict, or list of records.
        target_storage: If provided, write the reproduced tensor here
                        (Zarr or COG path).

    Returns:
        Reproduced :class:`GeoTensor`.

    Raises:
        ValueError: If the graph is empty or the first node is not
                    a load operation.
        FileNotFoundError: If the source data file is not accessible.
        NotImplementedError: If an operation in the graph is not
                             supported for replay.
    """
    if isinstance(provenance_data, str) and os.path.exists(provenance_data):
        with open(provenance_data, 'r') as f:
            data = json.load(f)
    elif isinstance(provenance_data, str):
        data = json.loads(provenance_data)
    else:
        data = provenance_data

    records = data.get('@graph', data) if isinstance(data, dict) else data

    if not records:
        raise ValueError("Provenance graph is empty — nothing to reproduce.")

    first = records[0] if isinstance(records[0], dict) else records[0].to_dict()
    source_path = first.get('parameters', {}).get('source') or \
                  first.get('parameters', {}).get('path')

    if not source_path:
        raise ValueError(
            "First provenance node must be a load operation with a "
            "'source' or 'path' parameter."
        )

    if not os.path.exists(source_path) and not source_path.startswith(('http://', 'https://', 's3://')):
        raise FileNotFoundError(
            f"Source data not found: {source_path}. "
            "Original data sources must be accessible for reproduction."
        )

    gt = _load_source(source_path, first.get('parameters', {}))

    _OP_DISPATCH = {
        'reproject': _replay_reproject,
        'resample': _replay_resample,
        'slice_bbox': _replay_slice_bbox,
        'resample_time': _replay_resample_time,
        'temporal_window': _replay_temporal_window,
        'temporal_aggregate': _replay_temporal_aggregate,
        'fill': _replay_fill,
        'convert_units': _replay_convert_units,
        'align': _replay_align,
    }

    for record in records[1:]:
        node = record if isinstance(record, dict) else record.to_dict()
        op = node.get('operation', '')
        params = node.get('parameters', {})

        if op in _OP_DISPATCH:
            gt = _OP_DISPATCH[op](gt, params)
        else:
            raise NotImplementedError(
                f"Operation '{op}' is not supported for replay. "
                f"Supported: {list(_OP_DISPATCH.keys())}"
            )

    if target_storage:
        if target_storage.endswith('.zarr'):
            from .backends import to_zarr
            to_zarr(gt, target_storage, compute=True)
        elif target_storage.endswith(('.tif', '.tiff', '.cog')):
            from .backends import to_cog
            to_cog(gt, target_storage)

    return gt


def _load_source(path: str, params: Dict[str, Any]) -> GeoTensor:
    """Load the initial data source for reproduction."""
    from .backends import open_cog, open_zarr, open_netcdf

    if path.endswith('.zarr'):
        return open_zarr(path)
    elif path.endswith('.nc'):
        return open_netcdf(path)
    else:
        band_info = params.get('band_info')
        chunks = params.get('chunks')
        return open_cog(path, band_info=band_info, chunks=chunks)


def _replay_reproject(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    target_crs = params.get('target_crs')
    resolution = params.get('resolution')
    method = params.get('method', 'bilinear')
    return gt.geotensor.reproject(target_crs, resolution=resolution, method=method)


def _replay_resample(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    resolution = params.get('resolution')
    method = params.get('method', 'bilinear')
    return gt.geotensor.resample(resolution=resolution, method=method)


def _replay_slice_bbox(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    west = params.get('west')
    south = params.get('south')
    east = params.get('east')
    north = params.get('north')
    crs = params.get('crs', 'EPSG:4326')
    return gt.geotensor.slice_bbox(west, south, east, north, crs=crs)


def _replay_resample_time(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    rule = params.get('rule', '1D')
    method = params.get('method', 'nearest')
    fill_value = params.get('fill_value', float('nan'))
    return gt.geotensor.resample_time(rule=rule, method=method, fill_value=fill_value)


def _replay_temporal_window(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    start = params.get('start')
    end = params.get('end')
    return gt.geotensor.temporal_window(start=start, end=end)


def _replay_temporal_aggregate(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    period = params.get('period', 'M')
    func = params.get('func', 'mean')
    return gt.geotensor.temporal_aggregate(period=period, func=func)


def _replay_fill(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    return gt.geotensor.fill(**params)


def _replay_convert_units(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    return gt.geotensor.convert_units(**params)


def _replay_align(gt: GeoTensor, params: Dict[str, Any]) -> GeoTensor:
    from .align import AlignmentSpec
    spec_data = params.get('spec')
    spec = AlignmentSpec() if not spec_data else AlignmentSpec(**spec_data)
    result = gt.geotensor.align(spec=spec)
    return result[0] if result else gt


def freeze(obj: Any) -> GeoTensor:
    """Freeze this GeoTensor, making it read-only.

    Any subsequent modifying operation raises
    :class:`ImmutableGeoTensorError`.

    Args:
        obj: DataArray / GeoTensor.

    Returns:
        The frozen :class:`GeoTensor` (self).
    """
    prov = get_provenance(obj)
    prov.frozen = True
    obj.attrs['provenance'] = prov
    return obj if isinstance(obj, GeoTensor) else to_geotensor(obj, obj)


def unfreeze(obj: Any) -> GeoTensor:
    """Unfreeze a previously frozen GeoTensor.

    Args:
        obj: Frozen DataArray / GeoTensor.

    Returns:
        The unfrozen :class:`GeoTensor`.
    """
    prov = get_provenance(obj)
    prov.frozen = False
    obj.attrs['provenance'] = prov
    return obj if isinstance(obj, GeoTensor) else to_geotensor(obj, obj)
