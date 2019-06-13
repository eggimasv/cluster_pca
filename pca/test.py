import geopandas as gpd
import osmnx as ox

from osgeo import ogr
import geopandas as gpd
import pandas as pd
from shapely import geometry
from shapely.geometry import box
from pyproj import Proj
from osgeo import ogr
import pandas as pd
import geopandas as gpd

ox.config(log_console=True, use_cache=True)
ox.__version__


# Define bounding box
bb  = [{
    'north': 46.89862993,
    'west': 8.2330575073,
    'south': 46.88889516,
    'east': 8.2465199857}]
bb = bb[0]
'''
# https://gis.stackexchange.com/questions/285336/convert-polygon-bounding-box-to-geodataframe
df_bb = pd.DataFrame.from_dict(bb)

# Get first entry
bb = bb[0]

#Get shapefile polygon of bbox
bb_polygon = box(bb['north'], bb['south'], bb['east'], bb['west'])

# Create geodataframe
gdf_bb = gpd.GeoDataFrame(df_bb, geometry=[bb_polygon])
print("Bounding Box Data")
print(gdf_bb)

# Assign coordinate system https://spatialreference.org/ref/epsg/wgs-84-utm-zone-32n/
gdf_bb.crs = {'init': 'epsg:32632'}

# convert crs
#gdf_bb_projected = ox.project_gdf(gdf_bb, to_crs={'proj':'longlat', 'epsg':'4326','ellps':'WGS84', 'datum':'WGS84'})

# Save a GeoDataFrame of place shapes or footprints as an ESRI shapefile.
#gdf_bb['gdf_name'] = 'test' # Assign name attribute
#ox.save_gdf_shapefile(gdf_bb)

fig, ax = ox.plot_shape(gdf_bb)

# Convert to shapely polygon
poly = geometry.Polygon([
        [bb['north'], bb['west']],
        [bb['north'], bb['east']],
        [bb['south'], bb['east']],
        [bb['south'], bb['west']]])




# create network from that bounding box
# https://wiki.openstreetmap.org/wiki/Key:network
# network type:   driving, walking, biking, route

# Return networkx graph
graph_osm = ox.graph_from_bbox(
    bb['north'],
    bb['south'],
    bb['east'],
    bb['west'],
    network_type='all_private')

# Project graph to UTM crs (#onvert from OSM coordinate system to UTM system)
graph_UTM = ox.project_graph(graph_osm) 

# Retrieve only edges from the graph
edges = ox.graph_to_gdfs(graph_UTM, nodes=False, edges=True)
nodes = edges = ox.graph_to_gdfs(graph_UTM, nodes=True, edges=False)

print("All columns of edges dataframe: {}".format(list(edges.columns))) # Get columns u: The first node of edge v: The last node of edge
#print(edges['maxspeed'].describe())
print("Current projection (crs): {}".format(edges.crs))

fig, ax = ox.plot_graph(graph_UTM)

'''

# Convert to shapely polygon
polygon = geometry.Polygon([
        [bb['north'], bb['west']],
        [bb['north'], bb['east']],
        [bb['south'], bb['east']],
        [bb['south'], bb['west']]])


from shapely.geometry import MultiPolygon
from shapely.geometry import shape
from collections import OrderedDict

multi = []
# append the geometries to the list
print("polygon")
print(polygon)

# Retrieve buildings from the area
#place = "Kamppi, Helsinki, Finland"
#buildings = ox.footprints.footprints_from_place(polygon, footprint_type='building', retain_invalid=False)
#print(buildings)


buildings = ox.footprints.create_footprints_gdf(
    north=bb['north'],
    south=bb['south'],
    east=bb['east'],
    west=bb['west'],
    footprint_type='building')

print("ff")

print(buildings)