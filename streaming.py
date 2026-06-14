"""STAC-native streaming engine for cloud-optimized geospatial data."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime as _datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class STACConfig:
    """Configuration for STAC streaming."""

    max_concurrent_requests: int = 4
    request_timeout: float = 30.0
    cache_dir: Optional[str] = None
    cache_size_gb: float = 10.0
    progressive_loading: bool = True

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="geoflow_stac_")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


class STACQuery:
    """Represents a STAC API query with spatial and temporal filters."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[_datetime, _datetime],
        collections: Optional[List[str]] = None,
        query_filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bbox = bbox
        self.time_range = time_range
        self.collections = collections or []
        self.query_filters = query_filters or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary for STAC API."""
        query: Dict[str, Any] = {
            "bbox": list(self.bbox),
            "datetime": f"{self.time_range[0].isoformat()}/{self.time_range[1].isoformat()}",
        }
        if self.collections:
            query["collections"] = self.collections
        if self.query_filters:
            query["filter"] = self.query_filters
        return query


class STACItem:
    """Represents a single STAC item with asset access."""

    def __init__(
        self,
        item_id: str,
        collection: str,
        bbox: Tuple[float, float, float, float],
        assets: Dict[str, Dict[str, Any]],
        properties: Dict[str, Any],
        datetime_value: Optional[_datetime] = None,
    ) -> None:
        self.item_id = item_id
        self.collection = collection
        self.bbox = bbox
        self.assets = assets
        self.properties = properties
        self.datetime = datetime_value

    def get_asset_href(self, asset_key: str = "image") -> Optional[str]:
        """Get the href URL for a specific asset."""
        asset = self.assets.get(asset_key)
        if asset:
            return asset.get("href")
        return None

    def get_geotiff_assets(self) -> List[str]:
        """Get all GeoTIFF asset hrefs."""
        hrefs = []
        for key, asset in self.assets.items():
            if asset.get("type") in ("image/tiff", "image/geotiff") or key.endswith((".tif", ".tiff")):
                hrefs.append(asset.get("href", ""))
        return hrefs


class STACStreamingEngine:
    """Cloud-native STAC integration for lazy loading and streaming.

    Features:
    - STAC query builder with spatial/temporal filters
    - Lazy remote chunk loading
    - Cloud-optimized geospatial streaming
    """

    def __init__(
        self,
        stac_api_url: str = "https://earth-search.aws.element84.com/v1",
        config: Optional[STACConfig] = None,
    ) -> None:
        self.stac_api_url = stac_api_url.rstrip("/")
        self.config = config or STACConfig()
        self._session = None

    def _get_session(self) -> Any:
        """Get or create a requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "GeoFlow/2.0.0"})
        return self._session

    def build_query(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[_datetime, _datetime],
        collections: Optional[List[str]] = None,
        query_filters: Optional[Dict[str, Any]] = None,
    ) -> STACQuery:
        """Build STAC query with spatial and temporal filters.

        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax).
            time_range: Time range (start, end).
            collections: STAC collection IDs.
            query_filters: Additional query filters.

        Returns:
            STACQuery object.
        """
        return STACQuery(
            bbox=bbox,
            time_range=time_range,
            collections=collections,
            query_filters=query_filters,
        )

    def search_items(
        self,
        query: STACQuery,
        max_items: int = 100,
    ) -> List[STACItem]:
        """Execute STAC search and return items.

        Args:
            query: STACQuery object.
            max_items: Maximum number of items to return.

        Returns:
            List of STACItem objects.
        """
        try:
            from pystac_client import Client
            client = Client.open(self.stac_api_url)

            search = client.search(
                bbox=query.bbox,
                datetime=f"{query.time_range[0].isoformat()}/{query.time_range[1].isoformat()}",
                collections=query.collections or None,
                max_items=max_items,
            )

            items = []
            for item in search.items():
                assets = dict(item.assets)
                items.append(STACItem(
                    item_id=item.id,
                    collection=item.collection_id if hasattr(item, "collection_id") else "",
                    bbox=item.bbox,
                    assets=assets,
                    properties=dict(item.properties) if hasattr(item, "properties") else {},
                    datetime=item.datetime,
                ))

            return items

        except ImportError:
            session = self._get_session()
            response = session.post(
                f"{self.stac_api_url}/search",
                json=query.to_dict(),
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            data = response.json()

            items = []
            for feature in data.get("features", [])[:max_items]:
                items.append(STACItem(
                    item_id=feature.get("id", ""),
                    collection=feature.get("collection", ""),
                    bbox=tuple(feature.get("bbox", [0, 0, 0, 0])),
                    assets=feature.get("assets", {}),
                    properties=feature.get("properties", {}),
                ))
            return items

    def lazy_load_chunk(
        self,
        asset_href: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_shape: Tuple[int, int] = (1024, 1024),
    ) -> Any:
        """Lazy load remote chunks with progressive loading.

        Args:
            asset_href: URL to the asset.
            bbox: Optional bounding box to load.
            target_shape: Target (height, width).

        Yields:
            GeoTensor chunks.
        """
        import rasterio
        from rasterio.windows import Window, from_bounds
        from .core import GeoTensor, to_geotensor

        with rasterio.open(asset_href) as src:
            if bbox:
                window = from_bounds(*bbox, transform=src.transform)
                window = window.round_offsets()
            else:
                window = Window(0, 0, src.width, src.height)

            chunk_h = min(target_shape[0], int(src.height))
            chunk_w = min(target_shape[1], int(src.width))

            for row_start in range(int(window.row_off), int(window.row_off + window.height), chunk_h):
                for col_start in range(int(window.col_off), int(window.col_off + window.width), chunk_w):
                    chunk_window = Window(
                        col_start, row_start,
                        min(chunk_w, int(window.col_off + window.width) - col_start),
                        min(chunk_h, int(window.row_off + window.height) - row_start),
                    )
                    data = src.read(window=chunk_window)
                    transform = src.window_transform(chunk_window)

                    from affine import Affine
                    gt = GeoTensor.from_array(
                        data,
                        transform=transform,
                        crs=src.crs,
                    )
                    yield gt

    def stream_cloud_optimized(
        self,
        asset_href: str,
        target_crs: Optional[str] = None,
        target_resolution: Optional[float] = None,
        normalization: str = "standard",
    ) -> Any:
        """Stream a Cloud-Optimized GeoTIFF with on-the-fly processing.

        Args:
            asset_href: URL to the COG.
            target_crs: Optional target CRS for reprojection.
            target_resolution: Optional target resolution.
            normalization: Normalization method.

        Returns:
            GeoTensor from the streamed COG.
        """
        import rasterio
        from .core import GeoTensor

        with rasterio.open(asset_href) as src:
            data = src.read()
            from affine import Affine
            gt = GeoTensor.from_array(
                data,
                transform=src.transform,
                crs=src.crs,
            )

        if target_crs:
            if hasattr(gt, 'rio'):
                import rioxarray  # noqa: F401
                gt = gt.rio.reproject(target_crs, resolution=target_resolution)
            else:
                from .align import align, AlignmentSpec, SpatialAlignment
                from .core import GeoTensor as GT
                ref = GeoTensor.from_array(
                    np.zeros((1, 1)),
                    transform=Affine(1, 0, 0, 0, -1, 0),
                    crs=target_crs,
                )
                aligned = align(gt, spec=AlignmentSpec(
                    spatial=SpatialAlignment(crs=target_crs, resolution=target_resolution),
                ))
                gt = aligned[0]

        if normalization == "standard":
            data = gt.values.astype(np.float32)
            mean = np.mean(data, axis=(-2, -1), keepdims=True)
            std = np.std(data, axis=(-2, -1), keepdims=True)
            std = np.clip(std, 1e-8, None)
            data = (data - mean) / std
            gt = GeoTensor(data, coords=gt.coords, dims=gt.dims, crs=gt.crs)

        return gt


def open_stac(
    url: str,
    collection: str,
    datetime_range: Optional[Union[str, Tuple[str, str]]] = None,
    datetime: Optional[Union[str, Tuple[str, str]]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    bands: Optional[List[str]] = None,
    max_cloud_cover: Optional[float] = None,
    max_items: int = 100,
) -> Any:
    """Open a STAC catalog and return a composite GeoTensor.

    Args:
        url: STAC API URL.
        collection: STAC collection ID.
        datetime_range: Time range as string ``"start/end"`` or tuple.
        datetime: Alias for ``datetime_range``. If both are provided,
                  ``datetime`` takes precedence.
        bbox: Bounding box ``(west, south, east, north)`` in EPSG:4326.
        bands: List of asset/band names to load.
        max_cloud_cover: Maximum cloud cover percentage filter.
        max_items: Maximum number of items to return.

    Returns:
        A GeoTensor with the requested data.
    """
    from .core import GeoTensor

    dt_range = datetime if datetime is not None else datetime_range

    if isinstance(dt_range, (tuple, list)):
        time_range = (_datetime.fromisoformat(dt_range[0]), _datetime.fromisoformat(dt_range[1]))
    elif isinstance(dt_range, str) and '/' in dt_range:
        parts = dt_range.split('/')
        time_range = (_datetime.fromisoformat(parts[0]), _datetime.fromisoformat(parts[1]))
    else:
        now = _datetime.now(timezone.utc)
        time_range = (now - timedelta(days=30), now)

    if bbox is None:
        bbox = (-180, -90, 180, 90)

    engine = STACStreamingEngine(stac_api_url=url)

    query_filter = {}
    if max_cloud_cover is not None:
        query_filter['eo:cloud_cover'] = {'lte': max_cloud_cover}

    query = engine.build_query(
        bbox=bbox,
        time_range=time_range,
        collections=[collection],
        query_filters=query_filter,
    )

    items = engine.search_items(query, max_items=max_items)
    if not items:
        raise ValueError(f"No STAC items found for collection '{collection}'.")

    item = items[0]
    if bands:
        tensors = []
        for band_name in bands:
            href = item.get_asset_href(band_name)
            if href:
                gt = engine.stream_cloud_optimized(href)
                tensors.append(gt)
        if not tensors:
            raise ValueError(f"No assets found for bands: {bands}")
        import xarray as xr
        combined = xr.concat(tensors, dim='band')
        return GeoTensor(
            combined.data,
            coords=combined.coords,
            dims=combined.dims,
            crs=tensors[0].crs,
        )

    geotiff_hrefs = item.get_geotiff_assets()
    if geotiff_hrefs:
        return engine.stream_cloud_optimized(geotiff_hrefs[0])

    raise ValueError(f"No usable assets found in STAC item '{item.item_id}'.")
