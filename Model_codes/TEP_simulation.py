import os
import pandas as pd
import numpy as np
import pyomo.environ as pyo

#Defining yearly transmission investment budget in USD
Tr_Budget = 4*10**9

#Defining the USD scaling factor to avoid numerical issues during optimization
Scalar = 10**4

#Defining negligible costs for transmission line flow and renewable energy
Min_Cost = 0.01/Scalar

#Defining unserved energy penalty
LOL_Penalty = 10000/Scalar

#Reading all necessary datasets
Generators_df = pd.read_csv("Inputs/data_genparams.csv", header=0)
Fuel_Prices_df = pd.read_csv("Inputs/Fuel_prices.csv", header=0)
Gen_Node_Matrix_df = pd.read_csv("Inputs/gen_mat.csv", header=0)
Hydro_max_df = pd.read_csv("Inputs/Hydro_max.csv", header=0)
Hydro_min_df = pd.read_csv("Inputs/Hydro_min.csv", header=0)
Mustrun_df = pd.read_csv("Inputs/must_run.csv", header=0)
Demand_df = pd.read_csv("Inputs/nodal_load.csv", header=0)
Solar_df = pd.read_csv("Inputs/nodal_solar.csv", header=0)
Wind_df = pd.read_csv("Inputs/nodal_wind.csv", header=0)
Offshore_Wind_df = pd.read_csv("Inputs/nodal_offshorewind.csv", header=0)
Line_Params_df = pd.read_csv("Inputs/line_params.csv", header=0)
Line_Node_Matrix_df = pd.read_csv("Inputs/line_to_bus.csv", header=0)
Line_Node_Matrix_df = Line_Node_Matrix_df.iloc[:,1:]

#Defining renewable and thermal generators
Renewable_Generators = Generators_df.loc[Generators_df['typ'].isin(['solar', 'wind','offshorewind','hydro'])].copy()
Thermal_Generators = Generators_df.loc[~Generators_df['typ'].isin(['solar', 'wind','offshorewind','hydro'])].copy()

#Reformatting generation-node matrix to exclude renewable generators
Gen_Node_Matrix_df = Gen_Node_Matrix_df.loc[Gen_Node_Matrix_df['name'].isin(Thermal_Generators['name'])]
Gen_Node_Matrix_df = Gen_Node_Matrix_df.iloc[:,1:]

#Defining the number of days in every month
Days_in_months = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
#Defining the number of hours in every month
Hours_in_months = Days_in_months*24

#Designating number of generators, nodes, lines and demand periods (months of the year = 12)
Num_Thermal_Gens = Thermal_Generators.shape[0]
Num_Nodes = Demand_df.shape[1]
Num_Lines = Line_Params_df.shape[0]
Num_Periods = len(Hours_in_months)

#Determining indices
Thermal_Gens = np.array([*range(0,Num_Thermal_Gens)]) 
Nodes = np.array([*range(0,Num_Nodes)])
Lines = np.array([*range(0,Num_Lines)])
Periods = np.array([*range(0,Num_Periods)])

#Saving important information as arrays for easy/fast referencing afterwards
Thermal_Gen_Names = Thermal_Generators["name"].values
Thermal_Gen_Types = Thermal_Generators["typ"].values
Thermal_Gen_Node = Thermal_Generators["node"].values
Thermal_Gen_MaxCap = Thermal_Generators["maxcap"].values
Thermal_Gen_HeatRate = Thermal_Generators["heat_rate"].values
Thermal_Gen_VarOM = Thermal_Generators["var_om"].values/Scalar

Thermal_Fuel_Prices = Fuel_Prices_df.values/Scalar
Gen_Node_Matrix = Gen_Node_Matrix_df.values

Node_Names = Demand_df.columns.values
Hydro_Max = Hydro_max_df.values
Hydro_Min = Hydro_min_df.values
Mustrun = Mustrun_df.values
Demand = Demand_df.values
Solar = Solar_df.values
Wind = Wind_df.values
Offshore_Wind = Offshore_Wind_df.values
Line_Node_Matrix = Line_Node_Matrix_df.values

Line_Names = Line_Params_df["line"].values
Line_Reactances = Line_Params_df["reactance"].values
Line_Initial_Limits = Line_Params_df["limit"].values
Line_Types = Line_Params_df["transmission_type"].values
Line_Lengths = Line_Params_df["length_mile"].values
Line_Costs = Line_Params_df["inv_cost_$_per_MWmile"].values/Scalar

############################ TRANSMISSION EXPANSION MODEL ############################

solver = "gurobi"

#Defining solver
opt = pyo.SolverFactory(solver)

#Creating empty arrays to store model results
ThermalGeneration_Results = np.zeros((Num_Periods, Num_Thermal_Gens))
SolarGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
WindGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
OffWindGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
HydroGeneration_Results = np.zeros((Num_Periods, Num_Nodes))
PowerFlow_Results = np.zeros((Num_Periods, Num_Lines))
UnservedEnergy_Results = np.zeros((Num_Periods, Num_Nodes))
VoltageAngle_Results = np.zeros((Num_Periods, Num_Nodes))
LineLimit_Results_np = np.zeros(Num_Lines)

#Initializing the model as concrete model
m=pyo.ConcreteModel()

#Telling model that we will need duals to check shadow prices for relevant constraints
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

#Defining sets
m.G=pyo.Set(initialize=Thermal_Gens) #All thermal generators
m.N=pyo.Set(initialize=Nodes) #All nodes
m.L=pyo.Set(initialize=Lines) #All transmission lines
m.T=pyo.Set(initialize=Periods) #All demand periods

#Defining decision variables
m.ThermalGen = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from thermal generators in MWh
m.SolarGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from solar generators in MWh
m.WindGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from wind generators in MWh
m.OffWindGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from offshore wind generators in MWh
m.HydroGen = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Generation from hydro generators in MWh

m.ActualFlow = pyo.Var(m.L, m.T, within=pyo.Reals, initialize=0) #Actual power flow on transmission lines in MWh
m.DummyFlow = pyo.Var(m.L, m.T, within=pyo.Reals, initialize=0) #Absolute value of power flow on transmission lines in MWh
m.NewLineCap = pyo.Var(m.L, within=pyo.PositiveReals) #New capacity of each transmission line in MW

m.LossOfLoad = pyo.Var(m.N, m.T, within=pyo.NonNegativeReals, initialize=0) #Unserved energy in MWh
m.VoltAngle = pyo.Var(m.N, m.T, within=pyo.Reals, bounds=(-180, 180)) #Voltage angle in degrees

#Defining objective function
def TEPCost(m):

    #Generation cost from dispatchable thermal generators
    Gen_cost = sum(m.ThermalGen[g,t]*((Thermal_Gen_HeatRate[g]*Thermal_Fuel_Prices[t,g])+Thermal_Gen_VarOM[g])*Hours_in_months[t] for g in m.G for t in m.T)
    #Loss of load (i.e., unserved energy cost)
    LOL_cost = sum(m.LossOfLoad[n,t]*LOL_Penalty*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from solar generators
    Solar_cost = sum(m.SolarGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from wind generators
    Wind_cost = sum(m.WindGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from offshore wind generators
    OffWind_cost = sum(m.OffWindGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Generation cost from hydro generators
    Hydro_cost = sum(m.HydroGen[n,t]*Min_Cost*Hours_in_months[t] for n in m.N for t in m.T)
    #Power flow cost on transmission lines
    Power_flow_cost = sum(m.DummyFlow[l,t]*Min_Cost*Hours_in_months[t] for l in m.L for t in m.T)
    #Investment cost for transmission capacity additions
    Tr_investment_cost = sum((m.NewLineCap[l]-Line_Initial_Limits[l])*Line_Lengths[l]*Line_Costs[l]/60 for l in m.L)

    return Gen_cost + LOL_cost + Solar_cost + Wind_cost + OffWind_cost + Hydro_cost + Power_flow_cost + Tr_investment_cost
    
m.ObjectiveFunc=pyo.Objective(rule=TEPCost, sense=pyo.minimize)

#Defining constraints

#Maximum capacity constraint for thermal generators
def MaxC(m,g,t):
    return m.ThermalGen[g,t] <= Thermal_Gen_MaxCap[g]
m.ThermalMaxCap_Cons= pyo.Constraint(m.G, m.T, rule=MaxC)

#Maximum capacity constraint for solar generators
def SolarMax(m,n,t):
    return m.SolarGen[n,t] <= Solar[t,n]
m.SolarMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=SolarMax)

#Maximum capacity constraint for wind generators
def WindMax(m,n,t):
    return m.WindGen[n,t] <= Wind[t,n]
m.WindMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=WindMax)

#Maximum capacity constraint for offshore wind generators
def OffWindMax(m,n,t):
    return m.OffWindGen[n,t] <= Offshore_Wind[t,n]
m.OffshoreWindMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=OffWindMax)

#Maximum capacity constraint for hydro generators
def HydroMax(m,n,t):
    return m.HydroGen[n,t] <= Hydro_Max[t,n]
m.HydroMaxCap_Cons= pyo.Constraint(m.N, m.T, rule=HydroMax)

#Minimum capacity constraint for hydro generators
def HydroMin(m,n,t):
    return m.HydroGen[n,t] >= Hydro_Min[t,n]
m.HydroMinCap_Cons= pyo.Constraint(m.N, m.T, rule=HydroMin)

#Voltage angle constraint for reference node
def ThetaRef(m,t):
    return m.VoltAngle[0,t] == 0
m.RefVoltAngle_Cons = pyo.Constraint(m.T, rule=ThetaRef)

#Kirchhoff's Voltage Law constraint
def KVL_Loop(m,l,t):
    theta_diff = sum(m.VoltAngle[n,t]*Line_Node_Matrix[l,n] for n in m.N)
    return  100*theta_diff == m.ActualFlow[l,t]*Line_Reactances[l]
m.KVLAroundLoopConstraint = pyo.Constraint(m.L, m.T, rule=KVL_Loop)

#Maximum transmission line flow constraint (positive side)
def FlowUP(m,l,t):
    return m.ActualFlow[l,t] <= m.NewLineCap[l]
m.FlowUP_Cons = pyo.Constraint(m.L, m.T ,rule=FlowUP)

#Maximum transmission line flow constraint (negative side)
def FlowDOWN(m,l,t):
    return -m.ActualFlow[l,t] <= m.NewLineCap[l]
m.FlowDOWN_Cons = pyo.Constraint(m.L, m.T ,rule=FlowDOWN)

#Minimum new line capacity constraint
def NewLCapacity(m,l):
    return Line_Initial_Limits[l] <= m.NewLineCap[l]
m.NewLCapacity_Cons = pyo.Constraint(m.L, rule=NewLCapacity)

#Constraint to find absolute value of flow (positive side)
def DummyFlowUP(m,l,t):
    return  m.DummyFlow[l,t] >= m.ActualFlow[l,t]
m.DummyFlowUP_Cons = pyo.Constraint(m.L, m.T ,rule=DummyFlowUP)

#Constraint to find absolute value of flow (negative side)
def DummyFlowDOWN(m,l,t):
    return  m.DummyFlow[l,t] >= -m.ActualFlow[l,t]
m.DummyFlowDOWN_Cons = pyo.Constraint(m.L, m.T ,rule=DummyFlowDOWN)

#Kirchhoff's Current Law (i.e., nodal power balance) constraint
def KCL(m,n,t):
    total_power_flow = sum(m.ActualFlow[l,t]*Line_Node_Matrix[l,n] for l in m.L)
    total_thermal_gen = sum(m.ThermalGen[g,t]*Gen_Node_Matrix[g,n] for g in m.G) 
    total_renewable_gen = m.SolarGen[n,t] + m.WindGen[n,t] + m.OffWindGen[n,t] + m.HydroGen[n,t]
    mustrun_gen = Mustrun[t,n]
    total_LOL = m.LossOfLoad[n,t]
    return total_thermal_gen + total_renewable_gen + total_LOL + mustrun_gen - total_power_flow == Demand[t,n]
m.KCL_Cons = pyo.Constraint(m.N, m.T, rule=KCL)

#Constraint to limit yearly transmission investment cost in USD
def TransmissionBudget(m):
    Tr_inv_cost = sum((m.NewLineCap[l]-Line_Initial_Limits[l])*Line_Lengths[l]*Line_Costs[l] for l in m.L)
    return Tr_inv_cost <= Tr_Budget/Scalar
m.TransmissionBudget_Cons = pyo.Constraint(rule=TransmissionBudget)

#Calling the solver to solve the model
TEP_results = opt.solve(m, tee=True)

#Checking the solver status and if solution is feasible or not
if (TEP_results.solver.status == pyo.SolverStatus.ok) and (TEP_results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\033[92mSuccess! Solution is feasible. \033[00m")
elif (TEP_results.solver.termination_condition == pyo.TerminationCondition.infeasible):
    print("\033[91mSolution is INFEASIBLE!!! \033[00m")
else:
    print(f"\033[91mSomething else is not right, solver status is {TEP_results.solver.status}. \033[00m")

#Creating a folder to store the results
os.makedirs('Outputs/TEP', exist_ok=True)

#Saving and writing thermal generation results
for g_n in Thermal_Gens:
    for t_n in Periods:
        ThermalGeneration_Results[t_n, g_n] = m.ThermalGen[g_n, t_n]()

ThermalGeneration_Results = pd.DataFrame(ThermalGeneration_Results, columns=Thermal_Gen_Names)
ThermalGeneration_Results.to_csv("Outputs/TEP/Thermal_Gen.csv", index=False)

#Saving and writing solar generation results
for n_n in Nodes:
    for t_n in Periods:
        SolarGeneration_Results[t_n, n_n] = m.SolarGen[n_n, t_n]()

SolarGeneration_Results = pd.DataFrame(SolarGeneration_Results, columns=Node_Names)
SolarGeneration_Results.to_csv("Outputs/TEP/Solar_Gen.csv", index=False)

#Saving and writing wind generation results
for n_n in Nodes:
    for t_n in Periods:
        WindGeneration_Results[t_n, n_n] = m.WindGen[n_n, t_n]()

WindGeneration_Results = pd.DataFrame(WindGeneration_Results, columns=Node_Names)
WindGeneration_Results.to_csv("Outputs/TEP/Wind_Gen.csv", index=False)

#Saving and writing offshore wind generation results
for n_n in Nodes:
    for t_n in Periods:
        OffWindGeneration_Results[t_n, n_n] = m.OffWindGen[n_n, t_n]()

OffWindGeneration_Results = pd.DataFrame(OffWindGeneration_Results, columns=Node_Names)
OffWindGeneration_Results.to_csv("Outputs/TEP/OffshoreWind_Gen.csv", index=False)

#Saving and writing hydro generation results
for n_n in Nodes:
    for t_n in Periods:
        HydroGeneration_Results[t_n, n_n] = m.HydroGen[n_n, t_n]()

HydroGeneration_Results = pd.DataFrame(HydroGeneration_Results, columns=Node_Names)
HydroGeneration_Results.to_csv("Outputs/TEP/Hydro_Gen.csv", index=False)

#Saving and writing unserved energy results
for n_n in Nodes:
    for t_n in Periods:
        UnservedEnergy_Results[t_n, n_n] = m.LossOfLoad[n_n, t_n]()

UnservedEnergy_Results = pd.DataFrame(UnservedEnergy_Results, columns=Node_Names)
UnservedEnergy_Results.to_csv("Outputs/TEP/Unserved_Energy.csv", index=False)

#Saving and writing voltage angle results
for n_n in Nodes:
    for t_n in Periods:
        VoltageAngle_Results[t_n, n_n] = m.VoltAngle[n_n, t_n]()

VoltageAngle_Results = pd.DataFrame(VoltageAngle_Results, columns=Node_Names)
VoltageAngle_Results.to_csv("Outputs/TEP/Voltage_Angle.csv", index=False)

#Saving and writing power flow results
for l_n in Lines:
    for t_n in Periods:
        PowerFlow_Results[t_n, l_n] = m.ActualFlow[l_n, t_n]()

PowerFlow_Results = pd.DataFrame(PowerFlow_Results, columns=Line_Names)
PowerFlow_Results.to_csv("Outputs/TEP/Power_Flow.csv", index=False)

#Saving and writing new transmission line limit results
for l_n in Lines:
    LineLimit_Results_np[l_n] = m.NewLineCap[l_n]()

LineLimit_Results = pd.DataFrame(LineLimit_Results_np, columns=["New_Capacity"])
LineLimit_Results.insert(0, "Name", Line_Names)
LineLimit_Results.insert(1, "Reactance", Line_Reactances)
LineLimit_Results.insert(2, "Type", Line_Types)
LineLimit_Results.insert(3, "Length", Line_Lengths)
LineLimit_Results.insert(4, "Capital_Cost", Line_Costs*Scalar)
LineLimit_Results.insert(5, "Old_Capacity", Line_Initial_Limits)
LineLimit_Results["Capacity_Addition"] = LineLimit_Results["New_Capacity"] - LineLimit_Results["Old_Capacity"]
LineLimit_Results["Total_Investment"] = LineLimit_Results["Capital_Cost"]*LineLimit_Results["Capacity_Addition"]*LineLimit_Results["Length"]
LineLimit_Results.to_csv("Outputs/TEP/New_Transmission_Data.csv", index=False)

GO_Line_File = pd.DataFrame(LineLimit_Results_np, columns=["limit"])
GO_Line_File.insert(0, "line", Line_Names)
GO_Line_File.insert(1, "reactance", Line_Reactances)
GO_Line_File.to_csv("Outputs/TEP/line_params.csv", index=False)

######################################################################################

