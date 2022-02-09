#!/usr/bin/env python
# coding: utf-8
try:
    import os
except ImportError as e:
    os.system('pip install os')
    import os
import utm
        
def download_osm(gdf):
    # Usage: nodes_gdf, segments_gdf = download_osm(gdf)
    # Download libraries
    import warnings
    warnings.filterwarnings("ignore")
    from shapely.geometry import Point, LineString
    import requests
    import pandas as pd
    
    from itertools import compress 
    try:
        import geopandas as gpd 
    except ImportError as e:
        os.system('pip install geopandas')
        import geopandas as gpd
    
    # Define the bounding box to query
    bounds = gdf.geometry.total_bounds

    # Build the query for overspass-api
    overpass_url = "http://overpass-api.de/api/interpreter"
#     overpass_query = """
#     [out:json];
#     (way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street"]
#     ["access"!~"private|no"]
#     ({0}, {1}, {2}, {3}););
#     out geom;
#     """.format(bounds[1], bounds[0], bounds[3], bounds[2])

    overpass_query = """
    [out:json];
    (way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street"]
    ({0}, {1}, {2}, {3}););
    out geom;
    """.format(bounds[1], bounds[0], bounds[3], bounds[2])

    # Query overpass-api
    response = requests.get(overpass_url, 
                            params={'data': overpass_query})

    # Put the response in a DataFrame
    data = response.json()
    ways_df = pd.DataFrame(data['elements'])

    # Parse the content in lists
    node_ids = []
    lat_lon = []
    way_ids = []
    oneway = []
    segment_seq = []

    n_nodes = [len(n) for n in list(ways_df.nodes)]

    [node_ids.extend(n) for n in list(ways_df.nodes)]
    [lat_lon.extend(g) for g in list(ways_df.geometry)]
    [way_ids.extend([ways_df.loc[i, 'id']]*n_nodes[i]) for i in range(0, len(ways_df))] 
    [oneway.extend([ways_df.loc[i, 'tags'].get('oneway', '0')]*n_nodes[i]) for i in range(0, len(ways_df))]
    [segment_seq.extend(list(range(1, n_nodes[i]+1))) for i in range(0, len(ways_df))] # segment sequence for that way_id

    # Convert to int to save memory
    oneway = [1 if s=='yes' else s for s in oneway] 
    oneway = [0 if s in ['no', '0', 'reversible', '-1', 'recommended'] else s for s in oneway] 
    oneway = [0 if s not in [0,'0',1,'1'] else s for s in oneway] 
    oneway = list(map(int, oneway))

    # ------------------------------------------------------------------------------------
    # ------------------------------ NODES -----------------------------------------------
    # ------------------------------------------------------------------------------------

    # Parse the json into a dataframe
    nodes = pd.DataFrame()
    nodes['way_id'] = way_ids
    nodes['node_id'] = node_ids
    nodes['oneway'] = oneway
    nodes['segment_seq'] = segment_seq

    # Get lat,lon values right
    lat = [p['lat'] for p in lat_lon]
    lon = [p['lon'] for p in lat_lon]

    # Create points
    points =  [Point(lon[i], lat[i]) for i in range(0, len(lat))]

    # Create GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(data=nodes, geometry = points)

    # ------------------------------------------------------------------------------------
    # --------------------------- SEGMENTS -----------------------------------------------
    # ------------------------------------------------------------------------------------

    # Define our lists
    # Does the node has the same way_id as the next node?
    bool_list = nodes['way_id'] == nodes['way_id'].shift(-1)
    # Nodes of the segment
    segment_nodes = ['{0} - {1}'.format(str(node_ids[i]), str(node_ids[i+1])) for i in range(0,len(node_ids)-1)]
    segment_ids = list(range(1, len(segment_nodes)+1))
    points_next = points[1:] + [None]

    # Remove the last node of the segment (it is already in the last segment)
    segment_nodes = list(compress(segment_nodes, bool_list)) 
    segment_ids = list(compress(segment_ids, bool_list)) 
    points = list(compress(points, bool_list)) 
    points_next = list(compress(points_next, bool_list)) 
    geometry = [LineString([points[i], points_next[i]]) for i in range(0,len(segment_nodes))]

    # Keep the segments and create the geo data frame
    segments = nodes.loc[bool_list, ['way_id', 'oneway', 'segment_seq']]
    segments['segment_nodes'] = segment_nodes
    segments['osm_segment_id'] = segment_ids
    segments_gdf = gpd.GeoDataFrame(data=segments, geometry = geometry)

    # ------------------------------------------------------------------------------------
    # --------------------------- ADD OPPOSITE SEGMENTS ----------------------------------
    # ------------------------------------------------------------------------------------

    # Create the opposite segments for two way streets
    opposite = segments_gdf.loc[segments_gdf.oneway == 0].reset_index()

    opp_nodes = ['{0} - {1}'.format(opposite.loc[i,'segment_nodes'].split(' - ')[1], opposite.loc[i,'segment_nodes'].split(' - ')[0]) for i in range(0,len(opposite))]
    opp_way_id = list(opposite.loc[:,'way_id'])
    opp_osm_segment_id = list(range(segments_gdf.osm_segment_id.max()+1, segments_gdf.osm_segment_id.max() + len(opposite) + 1))

    opp_geom = opposite.geometry.apply(lambda x: LineString([x.coords[1], x.coords[0]]))

    opp_df = pd.DataFrame()
    opp_df['way_id'] = opp_way_id
    opp_df['segment_nodes'] = opp_nodes
    opp_df['oneway'] = 0
    opp_df['osm_segment_id'] = opp_osm_segment_id
    opp_df['segment_seq'] = 0

    opp_gdf = gpd.GeoDataFrame(data=opp_df, geometry=opp_geom)

    segments_gdf = segments_gdf.append(opp_gdf)

    # Add "from" and "to" columns to make the graph generation easier
    segments_gdf['from'] = [int(s.split(' - ')[0]) for s in segments_gdf['segment_nodes']]
    segments_gdf['to'] = [int(s.split(' - ')[1]) for s in segments_gdf['segment_nodes']]
    
    return nodes_gdf, segments_gdf

def create_network(nodes_gdf, segments_gdf):
    # Usage: map_con = create_network(nodes_gdf, segments_gdf)
    # Download libraries
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from leuvenmapmatching.map.inmem import InMemMap 
    except ImportError as e:
        os.system('pip install leuvenmapmatching')
        from leuvenmapmatching.map.inmem import InMemMap 
    
    # Create the complete graph for all the network
    # Takes a long while, better do in advance
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)

    # Add nodes
    nodes_gdf.apply(lambda x: map_con.add_node(x.node_id, (x.geometry.coords[0][1], x.geometry.coords[0][0])), axis=1)

    # Add segments
    segments_gdf.apply(lambda x: map_con.add_edge(x['from'], x['to']), axis=1)
    
    return map_con
    
def buffer_shape(gdf, size_m):
    try:
        import utm
    except ImportError as e:
        os.system('pip install utm')
        import utm
    try:
        import geopandas as gpd 
    except ImportError as e:
        os.system('pip install geopandas')
        import geopandas as gpd
        
    try:
        from geopandas import GeoDataFrame 
    except ImportError as e:
        os.system('pip install geopandas')
        from geopandas import GeoDataFrame
        
    gdf.index=list(range(0,len(gdf)))
    gdf.crs = {'init':'epsg:4326'}
    lat_referece = gdf.geometry[0].coords[0][1]
    lon_reference = gdf.geometry[0].coords[0][0]

    zone = utm.from_latlon(lat_referece, lon_reference)
    #The EPSG code is 32600+zone for positive latitudes and 32700+zone for negatives.
    if lat_referece <0:
        epsg_code = 32700 + zone[2]
    else:
        epsg_code = 32600 + zone[2]

    # Create the buffers for the bus lines
    # At this point each segment has a polygon (the buffer)
    buffer_size = size_m
    buffers = GeoDataFrame(data = gdf.drop('geometry', axis=1), geometry = gdf.to_crs(epsg=epsg_code).buffer(buffer_size).to_crs(epsg=4326))
    
    return buffers
    
def code(gdf):
    try:
        import utm
    except ImportError as e:
        os.system('pip install utm')
        import utm
        
    try:
        import geopandas as gpd 
    except ImportError as e:
        os.system('pip install geopandas')
        import geopandas as gpd
        
    gdf.index=list(range(0,len(gdf)))
    gdf.crs = {'init':'epsg:4326'}
    try:
        # In case it is a Point or LineString
        lat_reference = gdf.geometry[0].coords[0][1]
        lon_reference = gdf.geometry[0].coords[0][0]
    except NotImplementedError:
        try:
            # In case it is a Polygon
            lat_reference = gdf.geometry[0].exterior.coords[0][1]
            lon_reference = gdf.geometry[0].exterior.coords[0][0]
        except AttributeError:
            # In case it is a Multipolygon
            lat_reference = gdf.geometry[0][0].exterior.coords[0][1]
            lon_reference = gdf.geometry[0][0].exterior.coords[0][0]

    zone = utm.from_latlon(lat_reference, lon_reference)
    #The EPSG code is 32600+zone for positive latitudes and 32700+zone for negatives.
    if lat_reference <0:
        epsg_code = 32700 + zone[2]
    else:
        epsg_code = 32600 + zone[2]
        
    return epsg_code

def format_shapes(s):
    import pandas as pd
    
    df = pd.DataFrame()
    for k in s.keys():
        df[k] = [s[k]]

    return df

def nearest_neighbor(gdf1, gdf2):
    import pandas as pd
    import os
    try:
        import geopandas as gpd
    except:
        os.system('pip install geopandas')
        import geopandas as gpd
        
    from shapely.geometry import Point, MultiPoint
    from shapely.ops import nearest_points
    
    geom1 = gdf1.geometry.name
    geom2 = gdf2.geometry.name
    
    destinations = MultiPoint(gdf2[geom2])
    
    nearest_list = gdf1[geom1].apply(lambda x: nearest_points(x, destinations))
    
    gdf1.rename(columns=dict(geometry='original_geom'), inplace=True)
    gdf1['geometry'] = [p[1] for p in nearest_list] 

    gdf1['geom_string'] = gdf1['geometry'].map(str)
    gdf2['geom_string'] = gdf2['geometry'].map(str)

    merged = pd.merge(gdf1, gdf2.drop('geometry', axis=1), left_on='geom_string', right_on='geom_string', how='left')
    merged = gpd.GeoDataFrame(data = merged.drop(['geometry', 'geom_string'], axis=1), geometry = merged.geometry)
    
    return merged

def match_to_shape(row, map_con, col_id, updates, log_match):
    # Usage: shape_net_seg = shapes.apply(lambda row: match_to_network(row, map_con, col_id), axis=1)
    # Usage(after first step): shape_net_seg_gdf = pd.concat([pd.DataFrame.from_dict(d) for d in shape_net_seg])
    # Download the libraries
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from leuvenmapmatching.matcher.distance import DistanceMatcher 
    except ImportError as e:
        os.system('pip install leuvenmapmatching')
        from leuvenmapmatching.matcher.distance import DistanceMatcher
        
    # Create the track to match to the street segments (graph above)
    # In this case I use a shape from the gtfs
    if col_id !='':
        shape_id = row[col_id]
    elif 'shape_id' in row.index:
        shape_id = row['shape_id']
    else:
        shape_id = row.name
    
    if updates:
        print('Working on shape_id {} which is number {} in the dataset'.format(shape_id, row.name))
    
    track = [(p[1], p[0]) for p in row.geometry.coords]
    max_dist = [40,60,100]
    dist = 100
    # Distances are in meters when using latitude-longitude.
    # map_con: Map object to connect to map database
    # obs_noise: Standard deviation of noise
    # obs_noise_ne: Standard deviation of noise for non-emitting states (is set to obs_noise if not give)
    # max_dist_init: Maximum distance from start location (if not given, uses max_dist)
    # max_dist: Maximum distance from path (this is a hard cut, min_prob_norm should be better)
    # min_prob_norm: Minimum normalized probability of observations (ema)
    # non_emitting_states: Allow non-emitting states. A non-emitting state is a state that is not associated with an observation. Here we assume it can be associated with a location in between two observations to allow for pruning. It is advised to set min_prob_norm and/or max_dist to avoid visiting all possible nodes in the graph.
    # max_lattice_width: Only keep track of this number of states (thus locations) for a given observation. Restrict the lattice (or possible candidate states per observation) to this value. If there are more possible next states, the states with the best likelihood so far are selected.
    # only_edges: Do not include nodes as states, only edges. This is the typical setting for HMM methods.
    # matching: Matching type
    # non_emitting_length_factor: Reduce the probability of a sequence of non-emitting states the longer it is. This can be used to prefer shorter paths. This is separate from the transition probabilities because transition probabilities are averaged for non-emitting states and thus the length is also averaged out.

    matcher = DistanceMatcher(map_con,
                              obs_noise=dist-25, obs_noise_ne=dist-25,  # meter
                              #max_dist_init=30, 
                              max_dist= dist, # meter
                              min_prob_norm=0.001, 
                              non_emitting_states=True,
                              max_lattice_width = 10, 
                              only_edges = True,
                              non_emitting_length_factor=.75,
                              dist_noise=dist,  # meter
                            )
    
    # Match the track to the graph
    try:
        states, lastidx = matcher.match(track)
        matched_segments = ['{} - {}'.format(st[0], st[1])  for st in states]
        # We now have the list of street segments the line follows in the list "matched_segments"
        # Now, we need to see to which bus segments each street segments matches.
        # We will overlay the street segments and the bus segments and matched them according to 
        # their ovelapping area

        
        if log_match:
            d= dict(shape_id=shape_id, network_segments = matched_segments, log_state='matched')
        else:
            d= dict(shape_id=shape_id, network_segments = matched_segments)
        
        return d
    
    except Exception as inst:
        if log_match:
            d = dict(shape_id=shape_id, network_segments = [], log_state = 'shape_id {} {}'.format(shape_id,str(inst).lower()))
        else:
            d = dict(shape_id=shape_id, network_segments = [])
        
        return d

def match_to_network(shapes, map_con, col_id='', updates=False, log_match=True):
    import pandas as pd

    shape_net_seg = shapes.apply(lambda row: match_to_shape(row, map_con, col_id, updates, log_match), axis=1)

    #shape_net_seg_gdf = pd.concat([pd.DataFrame.from_dict(d) for d in shape_net_seg])
    shape_net_seg_gdf = pd.concat([format_shapes(s) for s in shape_net_seg])
    shape_net_seg_gdf.reset_index(inplace=True)
    shape_net_seg_gdf.drop('index', axis=1, inplace=True)
        
    return shape_net_seg_gdf

def match_id_int(row, network, shape_net_seg_gdf, bsegments_gdf, updates): 
    from conflation import buffer_shape, code
    from shapely.geometry import MultiLineString
    import pandas as pd
    
    try:
        import geopandas as gpd 
    except ImportError as e:
        os.system('pip install geopandas')
        import geopandas as gpd
        
    shape_id = row.shape_id # returns a string
    
    # Get the bus segments
    b_segments = bsegments_gdf.loc[bsegments_gdf.shape_id==shape_id, ['segment_id', 'geometry']].reset_index().drop('index', axis=1)

    # Get the street segments
    if len(shape_net_seg_gdf.loc[shape_net_seg_gdf.shape_id==shape_id, 'network_segments'])>0:
        matched_segments = shape_net_seg_gdf.loc[shape_net_seg_gdf.shape_id==shape_id, 'network_segments'].values[0] # returns a list
    else:
        matched_segments=[]
    
    if len(matched_segments)>0:
        network_filtered = network.loc[network.segment_nodes.isin(matched_segments)].reset_index()

        # degrees*111139 = metres
        path_network = MultiLineString(list(network_filtered.geometry))
        path_bus = row.geometry
        max_dist = int(path_bus.hausdorff_distance(path_network)*111139 + 10) # max distance between paths
        buffer_size = max_dist/2

        check = False

        # Buffer both lists of segments
        buffers_bus = buffer_shape(b_segments, buffer_size)
        buffers_network = buffer_shape(network_filtered, buffer_size)
        # Overlay their buffers
        candidates = gpd.overlay(buffers_network, buffers_bus, how='intersection')

        # Check that every street segment matched to a bus segment
        # If they don't match they won't have a variable afterwards (speed, freq, etc.)
        check = len(buffers_network.loc[~buffers_network.osm_segment_id.isin(candidates.osm_segment_id.unique())]) == 0
        
        if check:
            #print('Shape {} succesfully matched with a {}m buffer'.format(shape_id, buffer_size))
            message = 'Shape {} succesfully matched with a {}m buffer'.format(shape_id, buffer_size)
        else: # Try with the max dist
            # Buffer both lists of segments
            buffers_bus = buffer_shape(b_segments, max_dist)
            buffers_network = buffer_shape(network_filtered, max_dist)
            # Overlay their buffers
            candidates = gpd.overlay(buffers_network, buffers_bus, how='intersection')
            check = len(buffers_network.loc[~buffers_network.osm_segment_id.isin(candidates.osm_segment_id.unique())]) == 0
            
            if check:
                #print('Shape {} succesfully matched with a {}m buffer'.format(shape_id, max_dist))
                message = 'Shape {} succesfully matched with a {}m buffer'.format(shape_id, max_dist)
            else:
                #print('Shape {} could not be matched to the netwrok with a {}m buffer'.format(shape_id, max_dist))
                message = 'Shape {} could not be matched to the netwrok with a {}m buffer'.format(shape_id, max_dist)
        
        if updates:
            print(message)
            
        # Some segments (mostly those in edges) can overlay with more than one bus segment
        # Let's keep only the match with the one the overlay the most.
        # Calculate the max overlay area for each street segment
        candidates['area'] = candidates.to_crs(epsg=code(network)).area.map(int)
        max_area = candidates.pivot_table('area',  index='osm_segment_id', aggfunc='max').rename(columns=dict(area='max_area'))
        candidates = pd.merge(candidates, max_area, left_on='osm_segment_id', right_index=True, how='left')

        # Filter only those with max intersection
        candidates = candidates.loc[candidates['area']==candidates['max_area']].drop_duplicates(subset=(['way_id', 'oneway', 'segment_seq', 'segment_nodes','osm_segment_id', 'from', 'to', 'area', 'max_area']))

        # Keep only the data we want
        candidates = candidates.loc[:, ['osm_segment_id','segment_nodes', 'segment_id']]
    else:
        candidates = pd.DataFrame()
        message = "Shape {} wasn't matched to the street network.".format(shape_id)
        
    d = dict(shape_id=shape_id, candidates = candidates)
    
    return  d
        
def match_id(shapes, segments, shape_net_seg_gdf, bsegments_gdf, updates=True):
    import pandas as pd
    
    shapes_ss = shapes.apply(lambda row: match_id_int(row, segments, shape_net_seg_gdf, bsegments_gdf, updates), axis=1)

    df = pd.concat([s['candidates'] for s in shapes_ss]).drop_duplicates()

    return df
    
def grid(gdf_in, size=200):
    import shapely
    try:
        import geopandas as gpd 
    except ImportError as e:
        os.system('pip install geopandas')
        import geopandas as gpd
    #from conflation import code

    gdf_proj = gdf_in.to_crs(epsg = code(gdf_in))
    sw = shapely.geometry.Point((gdf_proj.geometry.x.min(), gdf_proj.geometry.y.min()))
    ne = shapely.geometry.Point((gdf_proj.geometry.x.max(), gdf_proj.geometry.y.max()))

    stepsize = size # 100 m grid step size

    # Iterate over 2D area
    gridpoints = []
    x = sw.x
    while x < ne.x:
        y = sw.y
        while y < ne.y:
          # Get the parameters to create the 100 m x 100 m  box
          minx = x - stepsize/2
          miny = y - stepsize/2
          maxx = x + stepsize/2
          maxy = y + stepsize/2

          # Create the box
          box = shapely.geometry.box(minx, miny, maxx, maxy) #minx, miny, maxx, maxy) 
          gridpoints.append(box)
          y += stepsize
        x += stepsize
    # Create a GeoDataFrame with the grid created above
    grid_id = list(range(0, len(gridpoints)))
    gdf_out = gpd.GeoDataFrame(data = grid_id, geometry=gridpoints)
    gdf_out.columns = ['index', 'geometry']

    # Change the projection to EPSG:4326
    gdf_out.crs = {'init':'epsg:' + str(code(gdf_in))}
    gdf_out = gdf_out.to_crs(epsg=4326)

    return gdf_out