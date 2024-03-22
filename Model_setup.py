import pandas as pd
import numpy as np
import os
from shutil import copy
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

################## Defining model parameters ##################

#GO parameters
Nodes_no = 125 #Number of nodes in GO model
Formulation = "simple" #Defining if GO uses LP or MILP
Transmission_scaling = 500 #Transmission scaling factor of GO
Hurdle_scaling = -100 #Hurdle rate scaling factor of GO
Climate_scenario = "rcp45hotter_ssp3" #TGW climate scenario 

###############################################################

#TEP parameters
TEP_year = 2015 #Year to make transmission plans
Interreginonal_TEP_penalty = 0 #Transmission investment cost penalty in % for the lines crossing transmission planning region boundaries
Line_len_security_scalar = 30 #Line length adder in % for security purposes

###############################################################

#Defining the number of days in every month
Days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]
#Defining the number of hours in every month
Hours_in_months = [m*24 for m in Days_in_months]

#Creating a simulation folder for TEP optimization
TEP_case_name = f'{Nodes_no}_{Formulation}_{Transmission_scaling}_{Hurdle_scaling}_{Climate_scenario}_{TEP_year}_{Interreginonal_TEP_penalty}'
path_1 = str(Path(f'{Path.cwd()}/Simulation_folders/TEP_{TEP_case_name}'))
os.makedirs(path_1, exist_ok=True)

path_2 = str(Path(f'{Path.cwd()}/Simulation_folders/TEP_{TEP_case_name}/Inputs'))
os.makedirs(path_2, exist_ok=True)

path_3 = str(Path(f'{Path.cwd()}/Simulation_folders/TEP_{TEP_case_name}/Outputs'))
os.makedirs(path_3, exist_ok=True)

#Reading and organizing generator data and copying to simulation folder
GO_case_name = f'{Nodes_no}_{Formulation}_{Transmission_scaling}_{Hurdle_scaling}_{TEP_year}_{Climate_scenario}'

gen_data = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/data_genparams.csv", header=0)
gen_data_filt = gen_data.loc[:,["name","typ","node","maxcap","heat_rate","var_om"]].copy()
gen_data_filt.to_csv(f"{path_2}/data_genparams.csv", index=False)

#Copying generator/node matrix to simulation folder
copy(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/gen_mat.csv", path_2)

#Reading and calculating average fuel price and hydropower for each month and copying to simulation folder
fuel_price = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/Fuel_prices.csv", header=0)
fuel_price_monthly = pd.DataFrame(np.zeros((len(Days_in_months),fuel_price.shape[1])), columns=fuel_price.columns)

hydro_min = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/Hydro_min.csv", header=0)
hydro_min_monthly = pd.DataFrame(np.zeros((len(Days_in_months),hydro_min.shape[1])), columns=hydro_min.columns)

hydro_max = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/Hydro_max.csv", header=0)
hydro_max_monthly = pd.DataFrame(np.zeros((len(Days_in_months),hydro_max.shape[1])), columns=hydro_max.columns)

day_counter = 0
idx = 0
for month_day in Days_in_months:

    if idx == 0:
        fuel_filt = fuel_price.loc[0:month_day-1,:].mean().values
        fuel_price_monthly.loc[idx,:] = fuel_filt

        hydro_min_filt = hydro_min.loc[0:month_day-1,:].mean().values
        hydro_min_monthly.loc[idx,:] = hydro_min_filt

        hydro_max_filt = hydro_max.loc[0:month_day-1,:].mean().values
        hydro_max_monthly.loc[idx,:] = hydro_max_filt

    else:
        fuel_filt = fuel_price.loc[day_counter:day_counter+month_day-1,:].mean().values
        fuel_price_monthly.loc[idx,:] = fuel_filt

        hydro_min_filt = hydro_min.loc[day_counter:day_counter+month_day-1,:].mean().values
        hydro_min_monthly.loc[idx,:] = hydro_min_filt

        hydro_max_filt = hydro_max.loc[day_counter:day_counter+month_day-1,:].mean().values
        hydro_max_monthly.loc[idx,:] = hydro_max_filt

    day_counter += month_day
    idx += 1

fuel_price_monthly.to_csv(f"{path_2}/Fuel_prices.csv", index=False)
hydro_min_monthly.to_csv(f"{path_2}/Hydro_min.csv", index=False)
hydro_max_monthly.to_csv(f"{path_2}/Hydro_max.csv", index=False)


#Reading and organizing nuclear data and copying to simulation folder
nuclear = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/must_run.csv", header=0)
nuclear_monthly = pd.DataFrame(np.zeros((len(Days_in_months),hydro_min.shape[1])), columns=hydro_min.columns)

for node in hydro_min.columns:

    try:
        nuclear_monthly.loc[:,node] = np.repeat(nuclear.loc[0,node],len(Days_in_months))

    except KeyError:
        nuclear_monthly.loc[:,node] = np.repeat(0,len(Days_in_months))

nuclear_monthly.to_csv(f"{path_2}/must_run.csv", index=False)

#Reading and calculating peak net demand, respective total demand, wind and solar for each month and copying to simulation folder
load = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/nodal_load.csv", header=0)
load_monthly = pd.DataFrame(np.zeros((len(Days_in_months),load.shape[1])), columns=load.columns)

offshore_wind = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/nodal_offshorewind.csv", header=0)
offshore_wind_monthly = pd.DataFrame(np.zeros((len(Days_in_months),offshore_wind.shape[1])), columns=offshore_wind.columns)

onshore_wind = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/nodal_wind.csv", header=0)
onshore_wind_monthly = pd.DataFrame(np.zeros((len(Days_in_months),onshore_wind.shape[1])), columns=onshore_wind.columns)

solar = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/nodal_solar.csv", header=0)
solar_monthly = pd.DataFrame(np.zeros((len(Days_in_months),solar.shape[1])), columns=solar.columns)

#Calculating nodal hourly net demand (i.e., demand - solar - wind - offshore wind)
net_demand = load - solar - onshore_wind - offshore_wind
net_demand[net_demand < 0] = 0
net_demand_total = net_demand.sum(axis=1)

hour_counter = 0
idx = 0
for month_hour in Hours_in_months:

    if idx == 0:
        loc_max_net_demand = net_demand_total.loc[0:month_hour-1].argmax()
        c_idx = net_demand_total.loc[0:month_hour-1].index[loc_max_net_demand]

        load_monthly.loc[idx,:] = load.loc[c_idx,:].values
        offshore_wind_monthly.loc[idx,:] = offshore_wind.loc[c_idx,:].values
        onshore_wind_monthly.loc[idx,:] = onshore_wind.loc[c_idx,:].values
        solar_monthly.loc[idx,:] = solar.loc[c_idx,:].values
    
    else:
        loc_max_net_demand = net_demand_total.loc[hour_counter:hour_counter+month_hour-1].argmax()
        c_idx = net_demand_total.loc[hour_counter:hour_counter+month_hour-1].index[loc_max_net_demand]  

        load_monthly.loc[idx,:] = load.loc[c_idx,:].values
        offshore_wind_monthly.loc[idx,:] = offshore_wind.loc[c_idx,:].values
        onshore_wind_monthly.loc[idx,:] = onshore_wind.loc[c_idx,:].values
        solar_monthly.loc[idx,:] = solar.loc[c_idx,:].values

    hour_counter += month_hour
    idx += 1

load_monthly.to_csv(f"{path_2}/nodal_load.csv", index=False)
offshore_wind_monthly.to_csv(f"{path_2}/nodal_offshorewind.csv", index=False)
onshore_wind_monthly.to_csv(f"{path_2}/nodal_wind.csv", index=False)
solar_monthly.to_csv(f"{path_2}/nodal_solar.csv", index=False)

#Copying line/node matrix to simulation folder
copy(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/line_to_bus.csv", path_2)

#Reading existing line parameters
existing_lines = pd.read_csv(f"Datasets/GO_data/Exp{GO_case_name}/Inputs/line_params.csv", header=0)

#Reading node data
all_node_IDs = [int(i[4:]) for i in load_monthly.columns]

Node_dataset = pd.read_csv('Datasets/Node_data/10k_Nodes.csv', header=0)
Node_dataset = Node_dataset.loc[Node_dataset["Number"].isin(all_node_IDs)]
Node_dataset.reset_index(inplace=True,drop=True)
geometry = [Point(xy) for xy in zip(Node_dataset['Substation Longitude'],Node_dataset['Substation Latitude'])]
nodes_gdf = gpd.GeoDataFrame(Node_dataset,crs='epsg:4326',geometry=geometry)
nodes_gdf = nodes_gdf.to_crs("EPSG:3395")
nodes_gdf['TPR'] = np.repeat('Unknown',len(nodes_gdf))

#Reading and transmission region shapefile
Transmission_gdf = gpd.read_file('Datasets/Transmission_Regions_Shapefile/Transmission_Regions_Subregions.shp')
Transmission_gdf = Transmission_gdf.to_crs("EPSG:3395")
WECC_tr_gdf = Transmission_gdf.loc[Transmission_gdf['NAME']=='WESTERN ELECTRICITY COORDINATING COUNCIL (WECC)'].copy()
WECC_tr_gdf.reset_index(inplace=True,drop=True)

#Finding and saving transmision planning region of every node
for NN in range(0,len(Node_dataset)):

    node_point = nodes_gdf.loc[NN,'geometry']
    TPR_idx = node_point.within(WECC_tr_gdf['geometry']).idxmax()
    TPR_name = WECC_tr_gdf.loc[TPR_idx,'SUBNAME']

    if TPR_name == 'CA-MX US':
        nodes_gdf.loc[NN,'TPR'] = 'CAISO'
    elif TPR_name == 'NWPP':
        nodes_gdf.loc[NN,'TPR'] = 'NorthernGrid'
    else:
        nodes_gdf.loc[NN,'TPR'] = 'WestConnect'

#Creating columns for types, lengths and costs of transmission lines
existing_lines['transmission_type'] = np.repeat('Unknown',len(existing_lines))
existing_lines['length_mile'] = np.repeat(-1.5,len(existing_lines))
existing_lines['inv_cost_$_per_MWmile'] = np.repeat(-1.5,len(existing_lines))

#Iterating over transmission lines to calculate types, lengths and costs
for LL in range(0,len(existing_lines)):

    line_name = existing_lines.loc[LL,'line']
    splitted_name = line_name.split('_')

    #Checking if the line crosses TPR borders or not
    first_TPR = nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['TPR'].values[0]
    second_TPR = nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['TPR'].values[0]

    if first_TPR == second_TPR:
        line_type = 'regional'
    else:
        line_type = 'interregional'

    existing_lines.loc[LL,'transmission_type'] = line_type

    #Calculating the length of line in miles
    first_coordinates = (nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['Substation Latitude'].values[0],
                         nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[1])]['Substation Longitude'].values[0])
    
    second_coordinates = (nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['Substation Latitude'].values[0],
                         nodes_gdf.loc[nodes_gdf['Number']==int(splitted_name[2])]['Substation Longitude'].values[0])
    
    #Adding line length adder to line lengths for security purposes
    line_length = geodesic(first_coordinates, second_coordinates).miles * (1+(Line_len_security_scalar/100))
    existing_lines.loc[LL,'length_mile'] = round(line_length, 3)

    #Calculating $/MW-mile cost of each transmission line
    if line_length < 300: #Equation is y = 1.6875*x + 2671.25

        if line_type == 'regional':
            line_cost = (1.6875*line_length) + 2671.25
        else:
            line_cost = ((1.6875*line_length) + 2671.25) * (1+(Interreginonal_TEP_penalty/100))

        existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

    elif line_length >= 300 and line_length < 500: #Equation is y = -6.7*x + 5130
        
        if line_type == 'regional':
            line_cost = (-6.7*line_length) + 5130
        else:
            line_cost = ((-6.7*line_length) + 5130) * (1+(Interreginonal_TEP_penalty/100))

        existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

    elif line_length >= 500 and line_length < 1000: #Equation is y = -0.55*x + 2045

        if line_type == 'regional':
            line_cost = (-0.55*line_length) + 2045
        else:
            line_cost = ((-0.55*line_length) + 2045) * (1+(Interreginonal_TEP_penalty/100))
        
        existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3)

    else: #Equation is y = 1495

        if line_type == 'regional':
            line_cost = 1495 
        else:
            line_cost = 1495 * (1+(Interreginonal_TEP_penalty/100))

        existing_lines.loc[LL,'inv_cost_$_per_MWmile'] = round(line_cost, 3) 

existing_lines.to_csv(f"{path_2}/line_params.csv", index=False)

#Copying ED simulation script to simulation folder
copy("Model_codes/ED_simulation.py", path_1)

#Copying TEP simulation script to simulation folder
copy("Model_codes/TEP_simulation.py", path_1)

#Printing the status of the model setup script
print('Model setup for transmission expansion model is finished.')
    
