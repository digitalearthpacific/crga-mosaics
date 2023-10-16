import datetime
from odc.algo import mask_cleanup
from xarray import DataArray, concat
import xarray as xr


def mask_clouds_s2(xr: DataArray) -> DataArray:
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11
    clouds = (xr == CLOUD_SHADOWS) | (xr == CLOUD_HIGH_PROBABILITY)
    clouds = mask_cleanup(clouds, [("opening", 2), ("dilation", 3)])
    return clouds

def harmonise_s2(data):
    """
    Harmonize new Sentinel-2 data to the old baseline.

    Parameters
    ----------
    data: xarray.DataArray
        A DataArray with four dimensions: time, band, y, x

    Returns
    -------
    harmonized: xarray.DataArray
        A DataArray with all values harmonized to the old
        processing baseline.
    """
    cutoff = datetime.datetime(2022, 1, 25)
    offset = 1000
    bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]

    old = data.sel(time=slice(cutoff))

    to_process = list(set(bands) & set(data.band.data.tolist()))
    new = data.sel(time=slice(cutoff, None)).drop_sel(band=to_process)

    new_harmonized = data.sel(time=slice(cutoff, None), band=to_process).clip(offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(band=data.band.data.tolist())
    return xr.concat([old, new], dim="time")


def mask_clouds_ls(xr: DataArray, dilate: bool = False) -> DataArray:
    # dilated cloud, cirrus, cloud, cloud shadow
    mask_bitfields = [1, 2, 3, 4]
    bitmask = 0
    for field in mask_bitfields:
        bitmask |= 1 << field

    cloud_mask = xr.sel(band="qa_pixel").astype("uint16") & bitmask != 0

    if dilate:
        # From Alex @ https://gist.github.com/alexgleith/d9ea655d4e55162e64fe2c9db84284e5
        cloud_mask = mask_cleanup(cloud_mask, [("opening", 2), ("dilation", 3)])
    return xr.where(~cloud_mask)