import warnings
warnings.filterwarnings("ignore")
import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio.features
import stackstac
import pystac_client
import planetary_computer
import xrspatial.multispectral as ms
from dask_gateway import GatewayCluster
from dask.distributed import LocalCluster
from dask.distributed import Client
import geopandas as gpd
import util
from pystac.extensions.projection import ProjectionExtension as proj
import rioxarray


year = "2017/2023"
resolution = 10
cloud_threshold = 90

roi = gpd.read_file("aoi_gifts.gpkg", layer="aoi")

#Run with 25% Cloud Threshold
#country_codes = roi.code.to_list()

#Run with 80% Cloud Threshold
country_codes =['NR', 'FM', 'GU', 'KI', 'MH', 'MP', 'NU', 'PN', 'PW', 'TK', 'TV', 'WF', 'WS']

cluster = GatewayCluster()  # Creates the Dask Scheduler. Might take a minute.
client = cluster.get_client()
cluster.adapt(minimum=10, maximum=50)
print(cluster.dashboard_link)

stac = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

for country_code in country_codes:
    print("Processing: " + country_code)
    
    coi = roi[roi["code"] == country_code]
    aoi = coi.dissolve().geometry
    bbox = rasterio.features.bounds(aoi)
        
    search = stac.search(
        bbox = bbox,
        datetime = year,
        collections = ["sentinel-2-l2a"],
        query = {"eo:cloud_cover": {"lt": cloud_threshold}},
    )
    items = search.item_collection()
    print(len(items))
    
    item = next(search.get_items())
    epsg = proj.ext(item).epsg
    data = (
        stackstac.stack(
            items,
            assets=["B04", "B03", "B02", "SCL"],  
            epsg=epsg,
            bounds_latlon=bbox,
            chunksize=4096,
            resolution=resolution,
        )
        .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata    
    )
    
    #haromise
    data = util.harmonise_s2(data)
    
    #mask clouds
    data = data.where(~util.mask_clouds_s2(data.sel(band="SCL")))
    data = data.drop_sel(band="SCL")
    
    data = data.persist()
    median = data.median(dim="time").compute()
    image = ms.true_color(median.sel(band="B04"), median.sel(band="B03"), median.sel(band="B02"), c=15, th=0.125)
    
    image = image.transpose("band", "y", "x").squeeze()
    year = "2023"
    image.rio.to_raster(f"output/{country_code}_{year}.png", driver="PNG")  
    image.rio.to_raster(f"output/{country_code}_{year}.tif", driver="GTiff")  
    

cluster.close()
print("Finished.")
