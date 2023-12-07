# 引用需要的套件
from datetime import *
from interval import Interval
import time
import configparser as cp
import pandas as pd
import numpy as np
import random
import csv
import os
from pathlib import Path
from datetime import datetime,date,time,timedelta
from sklearn.cluster import KMeans
import requests,json

import gurobipy as gp
from gurobipy import GRB

class OnGrid_realtime_opt:
    def __init__(self, mins : int):
        # 外部參數
        self.mins = mins
        # 時間單位長度
        self.start_time = 0
        self.end_time = 24
        self.time_interval = int(self.end_time - self.start_time)
        self.time_scale = int(60/self.mins)
        self.time = [i for i in range(self.time_interval*self.time_scale)] 
        # 最佳化控制長度
        self.len_n = range(0,self.time_interval*self.time_scale)
        # 時間概念長度
        self.time_n = range(self.start_time*self.time_scale,self.end_time*self.time_scale)
        # 最佳化控制頻率
        self.freq = str(self.mins)+"min"
    def realtime_opt_parameter(self):
        # 系統參數               
        self.read_opt_parameter()
        # 儲能系統參數
        self.read_ESS_parameter()
        # 規劃場域參數
        self.read_planning_area_data()
        # 輔助服務費用及參數
        self.read_energytrading_info()
        # 七櫃淨負載分群資料
        self.netload_history_data()
        # 各不確定性機率
        self.uncertainty_probability()
        # 時間電價               
        self.TOU_scenario_ini()
        # 最佳化排程結果
        self.opt_result_record_data()
    def read_opt_parameter(self): 
        self.planning_year = 30
        self.discount_rate = 1.875/100
        self.spr_status = 1     # 1:參與即時備轉 0:不參與即時備轉
    def read_ESS_parameter(self): 
        self.battery_ub = 12
        self.battery_lb = 8
        self.pcs_ub = 12
        self.pcs_lb = 8
        self.ESS_SOCmax = 90/100
        self.ESS_SOCmin = 10/100
        self.ESS_SOCinitial = self.ESS_SOCmin
        self.ess_calender_life = 10
        self.mESS = round(1/6000,7)
        self.Ess_efficiency = 95/100
        self.ess_cost = 16000                                                             #電池建置價格
        self.pcs_cost = 3500                                                              #pcs建置價格
    def read_planning_area_data(self):
        self.cc_upper = 12
        self.Ccap_base = 160.6/30
        self.planning_area_cc = 15
    def netload_history_data(self):
        os.chdir(r'D:\EPSLab_pc\Thomas_碩論\廠商提供')
        self.netload_scenario_data = pd.read_csv('netload_21_scaleup_ClusterCenters.csv')
        self.netload_scenario_power = self.netload_scenario_data.iloc[:,1:]
        self.netload_scenario_nums = self.netload_scenario_data.shape[0]
        self.netload_scenario_days = self.netload_scenario_data['days']
    def TOU_scenario_ini(self):
        os.chdir(r'D:\EPSLab_pc\Thomas_碩論\廠商提供')
        tou_data = pd.read_csv('tou_data.csv')
        nonsummer_tou = np.array(tou_data.iloc[0, 1:])
        summer_tou = np.array(tou_data.iloc[1, 1:])
        self.TOU_scenario = np.array([np.repeat(nonsummer_tou, self.time_scale)/self.time_scale,np.repeat(summer_tou, self.time_scale)/self.time_scale])
    def read_energytrading_info(self):
        self.SpR_cap_price = np.array([280/self.time_scale]*(self.time_scale*self.time_interval))
        self.SpR_eff_price = np.array([100/self.time_scale]*(self.time_scale*self.time_interval))
        self.SpR_energy_price = np.array([2700/1000/self.time_scale]*(self.time_scale*self.time_interval))
        self.SpR_exe_times = 35
        start_time = "17:30:00"
        self.SpR_start_time = round(int(str(start_time)[0:2])*self.time_scale+int(str(start_time)[3:5])/self.mins)
    def opt_result_record_data(self):
        # 儲存最佳化的參數
        self.netload = np.array([[0.0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)])
        self.PESS = np.array([[[[0.0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.ESS_SOC = np.array([[[[0.0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.Pnet = np.array([[[[0.0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.Pnet_cost = np.array([[[[0.0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.SpR_energy_price_record = np.array([[[[0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.SpR_energy_execution_record = np.array([[[[0 for i in range(96)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.sbspm = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.Cap_SpR = []
        # 即時備轉基準容量
        self.SpR_CBL_base = np.array([[[[0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # 最佳化成本計算
        self.optimize_cost_data = pd.DataFrame()
    def uncertainty_probability(self):
        self.netload_uncer_prob = self.netload_scenario_days
        if(self.spr_status == 0):
            self.ASS_time = [0]
            self.ASS_time_uncer_prob = [1]
        else:
            self.ASS_time = [-1,int(13*self.time_scale),int(14*self.time_scale),int(15*self.time_scale),int(16*self.time_scale),int(17*self.time_scale),int(18*self.time_scale)]
            self.ASS_time_uncer_prob = [(365-35)/365, (35/365)*0.03, (35/365)*0.06, (35/365)*0.12, (35/365)*0.15, (35/365)*0.54, (35/365)*0.1]
        # self.ASS_time = [0, 1]
        # self.ASS_time_uncer_prob = [(365-self.SpR_exe_times)/365, self.SpR_exe_times/365]
        self.tou_uncer_prob = [212/365,153/365]
    def centralize_optimization(self):
        Y = self.planning_year
        ESS_calender_life = self.ess_calender_life
        r = self.discount_rate
        ASS_status = self.spr_status
        # ----------------MILP Algorithm---------------------
        Cap_SpR_scale = 0.25
        Cap_ESS_scale = 0.25
        Pcap_scale = 0.25
        # 最佳化排程長度
        len_n = self.len_n
        # 即時備轉容量費與效能費
        SpR_eff_and_cap_price = []
        for i in len_n:
            SpR_eff_and_cap_price.append(self.SpR_eff_price[i] + self.SpR_cap_price[i])
        # 最佳化排程
        m = gp.Model()
        # variable
        # net cost
        Pnet_cost = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        Ccap1 = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n]for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        Ccap2 = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        MAX_Ccap1 = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        MAX_Ccap2 = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # net
        Pnet = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # ESS
        Pess_c = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        Pess_dsc = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        Uess_c = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        Uess_dsc = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SOC_ESS = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # ASS or DRB
        U_SpR_100 = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        U_SpR_95 = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        U_SpR_85 = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        U_SpR_70 = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_100_quality_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_95_quality_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_85_quality_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_70_quality_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_100_quality_total_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_95_quality_total_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_85_quality_total_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_70_quality_total_execution = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_energy_execution = np.array([[[[m.addVar(lb=-float('inf'),vtype=GRB.CONTINUOUS) for i in len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_CBL_base = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        SpR_sbspm = np.array([[[[m.addVar(lb=0,vtype=GRB.CONTINUOUS)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # SpR投標容量
        Cap_SpR = [m.addVar(ub=self.pcs_ub/Cap_SpR_scale,vtype=GRB.INTEGER) for i in range(self.time_interval)]
        Cap_SpR_rec = [m.addVar(lb=0,vtype=GRB.CONTINUOUS) for i in range(self.time_interval)]
        # ESS build
        battery_cap = m.addVar(lb=self.battery_lb/Cap_ESS_scale,ub=self.battery_ub/Cap_ESS_scale,vtype=GRB.INTEGER)                                                                                    #ESS儲能容量
        Cap_ESS_rec = m.addVar(lb=0,vtype=GRB.CONTINUOUS)                                                                                     
        pcs_cap = m.addVar(lb=self.pcs_lb/Cap_ESS_scale,ub=self.pcs_ub/Cap_ESS_scale,vtype=GRB.INTEGER)                                                                                    #ESS儲能功率
        ESS_initial_cost = m.addVar(lb=0,vtype=GRB.CONTINUOUS)  
        EssSOC_c = m.addVar(lb=0,vtype=GRB.CONTINUOUS)
        EssSOC_dsc = m.addVar(lb=0,vtype=GRB.CONTINUOUS)
        Essdeg_c = m.addVar(lb=0,vtype=GRB.CONTINUOUS)
        Essdeg_dsc = m.addVar(lb=0,vtype=GRB.CONTINUOUS)
        # 執行率不佳次數限制
        U_SpR_fail = np.array([[[[m.addVar(lb=0,ub=1,vtype=GRB.BINARY)] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        # Big M
        M = 1000000000
        # 契約容量
        Contr_cap = m.addVar(ub=self.cc_upper/Pcap_scale,vtype=GRB.INTEGER)

        # 目標函式
        total_Pnet_cost, total_SpR_energy_execution, total_SpR_100_execution, total_SpR_95_execution, total_SpR_85_execution, total_SpR_70_execution, total_Ccap_cost, total_Ccap1_cost, total_Ccap2_cost, total_Cap_SpR_cost = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ESS_replace_cost = 0
        for y in range(1, Y):
            # 電網購電成本
            total_Pnet_cost += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Pnet_cost[TOU_scenario][ASS_day_scenario][netload_scenario][i] for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
            if(ASS_status == 1):
                # MGO即時備轉電能費
                total_SpR_energy_execution += (1/(1+r)**(y))*gp.quicksum(-self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * SpR_energy_execution[TOU_scenario][ASS_day_scenario][netload_scenario][i] for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
                # MGO即時備轉服務品質費用
                total_SpR_100_execution += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * SpR_100_quality_total_execution[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
                total_SpR_95_execution += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * SpR_95_quality_total_execution[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
                total_SpR_85_execution += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * SpR_85_quality_total_execution[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
                total_SpR_70_execution += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * SpR_70_quality_total_execution[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
                # 投標容量費
                total_Cap_SpR_cost += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Cap_SpR[int(i/4)] * Cap_SpR_scale * self.SpR_cap_price[i] for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))) 
            # 契約容量費
            total_Ccap_cost += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Contr_cap*1000 * Pcap_scale * self.Ccap_base for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
            # 契約容量超約成本
            total_Ccap1_cost += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * MAX_Ccap1[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
            total_Ccap2_cost += (1/(1+r)**(y))*gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * MAX_Ccap2[TOU_scenario][ASS_day_scenario][netload_scenario][0] for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob)))
            # 替換成本
            if(y != Y and y%ESS_calender_life == 0):
                print('will replacement',y)
                ESS_replace_cost += (1/(1+r)**y)*ESS_initial_cost
        # Objective function
        m.setObjective(total_Pnet_cost                                                                                  # 電網購電成本
                    +total_SpR_energy_execution                                                                         # MGO即時備轉電能費
                    +total_SpR_100_execution                                                                            # MGO即時備轉服務品質100費用
                    +total_SpR_95_execution                                                                             # MGO即時備轉服務品質95費用
                    +total_SpR_85_execution                                                                             # MGO即時備轉服務品質85費用
                    +total_SpR_70_execution                                                                             # MGO即時備轉服務品質70費用
                    +total_Ccap_cost                                                                                    # 契約容量費
                    +total_Ccap1_cost+total_Ccap2_cost                                                                  # 契約容量超約成本
                    -total_Cap_SpR_cost                                                                                 # 投標容量費
                    +ESS_initial_cost
                    +ESS_replace_cost, sense=GRB.MINIMIZE)      
        # constraint
        # 計算儲能係數
        m.addConstr(EssSOC_c == self.Ess_efficiency*Cap_ESS_rec/(self.time_scale), "ess_sizing_1")                                                  #ESS_c=det(t)*efficiency/Cap_ESS
        m.addConstr(EssSOC_dsc == Cap_ESS_rec/((self.time_scale)*self.Ess_efficiency), "ess_sizing_2")                                              #ESS_dsc=det(t)/(Cap_ESS*efficiency)
        # m.addConstr(Essdeg_c == (ESS_initial_cost/2)*Cap_ESS_rec*self.mESS/(self.time_scale)*self.Ess_efficiency, "c03")                  #(建置成本/2)/(容量*總充放次數)*(det(t)*效率轉換)
        # m.addConstr(Essdeg_dsc == (ESS_initial_cost/2)*Cap_ESS_rec*self.mESS/(self.time_scale)/self.Ess_efficiency, "c04")                #(建置成本/2)/(容量*總充放次數)*(det(t)*效率轉換)
        m.addConstr(ESS_initial_cost == self.ess_cost*battery_cap*1000*Cap_ESS_scale + self.pcs_cost * pcs_cap*1000*Cap_ESS_scale, "ess_sizing_3")     #ESS儲能建置成本(容量*單位容量下的價格)
        m.addConstr(battery_cap*1000*Cap_ESS_scale * Cap_ESS_rec== 1, "ess_sizing_4")
        m.addConstr(pcs_cap*Cap_ESS_scale <= battery_cap*Cap_ESS_scale, "ess_sizing_5")
        # 電力平衡限制式
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(self.netload_scenario_power.iloc[netload_scenario,i] + Pess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i] - Pess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i] == Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i],"equilibrium")
        # 變壓器限制式
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= 0, "transformer")
        # 電價限制式
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(Pnet_cost[TOU_scenario][ASS_time_scenario][netload_scenario][i] == Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] * self.TOU_scenario[TOU_scenario][i], "Pnet_cost")
        # 契約容量
        # 10% 內
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= 0, "Ccap1_1")
                        m.addConstr(Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= (Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] - Contr_cap*1000 * Pcap_scale) * (self.Ccap_base*2), "Ccap1_2")
                        m.addConstr(MAX_Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][i], "Ccap1_3")
        # 10% 外
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= 0, "Ccap2_1")
                        m.addConstr(Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= (Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] - Contr_cap*1000 * Pcap_scale * 1.1) * (self.Ccap_base), "Ccap2_2")
                        m.addConstr(MAX_Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][i], "Ccap2_3")
        # 儲能功率限制式
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                    for i in len_n:
                        m.addConstr(Pess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i] <= pcs_cap*1000*Cap_ESS_scale * Uess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i], "Pess_1")
                        m.addConstr(Pess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i] <= pcs_cap*1000*Cap_ESS_scale * Uess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i], "Pess_2")
                        m.addConstr(Uess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i] + Uess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i] <= 1, "Pess_3")
        # ESS SOC 限制式
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):     
                    for i in len_n:
                        if i == 0:
                            m.addConstr(SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i] == self.ESS_SOCinitial + EssSOC_c * Pess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i] - EssSOC_dsc * Pess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i], "ESS_soc_1")
                        else:
                            m.addConstr(SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i] == SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i-1] + EssSOC_c * Pess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i] - EssSOC_dsc * Pess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i], "ESS_soc_2")
                        m.addConstr(SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i] <= self.ESS_SOCmax, "ESS_soc_3")
                        m.addConstr(SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= self.ESS_SOCmin, "ESS_soc_4")
                    # 儲能結束時間SOC等於開始時間SOC
                    m.addConstr(SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][int(self.time_interval*self.time_scale-1)] == self.ESS_SOCmin, "ESS_soc_5")
        if(ASS_status == 1):
            # 即時備轉投標量
            for i in range(self.time_interval):
                m.addConstr(Cap_SpR_rec[i] * Cap_SpR[i] * Cap_SpR_scale == 1, "SpR_cap_1")
                m.addConstr(Cap_SpR[i] * Cap_SpR_scale <= pcs_cap*Cap_ESS_scale, "SpR_cap_2")
                m.addConstr(Cap_SpR[i] * Cap_SpR_scale <= battery_cap*Cap_ESS_scale, "SpR_cap_3")
                m.addConstr(Cap_SpR[i] * Cap_SpR_scale * 1000 <= gp.quicksum(self.netload_scenario_days[netload_scenario]/365 * sum(self.netload_scenario_power.iloc[netload_scenario,i*4:i*4+4])/4 for netload_scenario in range(self.netload_scenario_nums)), "SpR_cap_4")
            # for i in range(self.SpR_start_time,self.SpR_start_time+4):
            #     m.addConstr(Cap_SpR[int(i/4)] == Cap_SpR[int((i+1)/4)], "SpR_cap_5")
            # CBL
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0:
                            m.addConstr(SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] == self.netload_scenario_power.iloc[netload_scenario, self.ASS_time[ASS_time_scenario]-1], "CBL_1")
                        else:
                            m.addConstr(SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] == 0, "CBL_2")
            # sbspm
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0:
                            m.addConstr(SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] == (SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - (gp.quicksum(Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][(self.ASS_time[ASS_time_scenario]):(self.ASS_time[ASS_time_scenario]+4)])/4))*Cap_SpR_rec[int(self.ASS_time[ASS_time_scenario]/4)]/1000, "sbspm_1")
                        else:
                            m.addConstr(SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] == 1, "sbspm_2")
            # 儲能確保有足夠電力交易量
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0:
                            for i in len_n:
                                # 即時備轉執行價金，績效: 做多少 / 要做多少 (SpR_CBL_base - Pnet[i]) / SpR_cap 
                                if (self.ASS_time[ASS_time_scenario] + self.time_scale) > i >= self.ASS_time[ASS_time_scenario]:
                                    # 未達到100%執行率，會觸發
                                    m.addConstr(1 - ((SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i]) * Cap_SpR_rec[int(i/4)]/1000) <= U_SpR_100[TOU_scenario][ASS_time_scenario][netload_scenario][i]*M, "U_spr_1")
                                    # 未達到95%，會觸發
                                    m.addConstr(0.95 - ((SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i]) * Cap_SpR_rec[int(i/4)]/1000) <= U_SpR_95[TOU_scenario][ASS_time_scenario][netload_scenario][i]*M, "U_spr_2")
                                    # 未達到80%，會觸發
                                    m.addConstr(0.85 - ((SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i]) * Cap_SpR_rec[int(i/4)]/1000) <= U_SpR_85[TOU_scenario][ASS_time_scenario][netload_scenario][i]*M, "U_spr_3")
                                    # 未達到70%，會觸發
                                    m.addConstr(0.7 - ((SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i]) * Cap_SpR_rec[int(i/4)]/1000) <= U_SpR_70[TOU_scenario][ASS_time_scenario][netload_scenario][i]*M, "U_spr_4")
                                else:
                                    m.addConstr(U_SpR_100[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_5")
                                    m.addConstr(U_SpR_95[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_6")
                                    m.addConstr(U_SpR_85[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_7")
                                    m.addConstr(U_SpR_70[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_8")
                        else:
                            for i in len_n:
                                m.addConstr(U_SpR_100[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_9")
                                m.addConstr(U_SpR_95[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_10")
                                m.addConstr(U_SpR_85[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_11")
                                m.addConstr(U_SpR_70[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "U_spr_12")
            # 即時備轉執行率 < 70% 次數不得超過3次
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        m.addConstr(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_time_scenario] * self.netload_scenario_days[netload_scenario] * (gp.quicksum(U_SpR_70[TOU_scenario][ASS_time_scenario][netload_scenario][i] for i in len_n)/4) == U_SpR_fail[TOU_scenario][ASS_time_scenario][netload_scenario][0], "U_SpR_fail")
            m.addConstr(gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * U_SpR_fail[TOU_scenario][ASS_day_scenario][netload_scenario][0] for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))) <= 3, "U_SpR_fail")
            # 即時備轉電能費
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0:
                            for i in len_n:
                                if (self.ASS_time[ASS_time_scenario] + self.time_scale) > i >= self.ASS_time[ASS_time_scenario]:
                                    m.addConstr(SpR_energy_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == (SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] - Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i]) * self.SpR_energy_price[i], "SpR_energy_execution_1")
                                else:
                                    m.addConstr(SpR_energy_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "SpR_energy_execution_2")
                        else:
                            for i in len_n:
                                m.addConstr(SpR_energy_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == 0, "SpR_energy_execution_3")
            # 即時備轉服務品質指標費用
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        for i in len_n:
                            n = 0.1
                            m.addConstr(SpR_100_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == U_SpR_100[TOU_scenario][ASS_time_scenario][netload_scenario][i] * SpR_eff_and_cap_price[i] * Cap_SpR[int(i/4)] * Cap_SpR_scale * n, "SpR_100_quality_execution")
                            m.addConstr(SpR_95_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == U_SpR_95[TOU_scenario][ASS_time_scenario][netload_scenario][i] * SpR_eff_and_cap_price[i] * Cap_SpR[int(i/4)] * Cap_SpR_scale * (0.3 - n), "SpR_95_quality_execution")
                            m.addConstr(SpR_85_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == U_SpR_85[TOU_scenario][ASS_time_scenario][netload_scenario][i] * SpR_eff_and_cap_price[i] * Cap_SpR[int(i/4)] * Cap_SpR_scale * 0.7, "SpR_85_quality_execution")
                            m.addConstr(SpR_70_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i] == U_SpR_70[TOU_scenario][ASS_time_scenario][netload_scenario][i] * SpR_eff_and_cap_price[i] * Cap_SpR[int(i/4)] * Cap_SpR_scale * 240, "SpR_70_quality_execution")
                            m.addConstr(SpR_100_quality_total_execution[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= SpR_100_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i], "SpR_100_quality_total_execution")
                            m.addConstr(SpR_95_quality_total_execution[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= SpR_95_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i], "SpR_95_quality_total_execution")
                            m.addConstr(SpR_85_quality_total_execution[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= SpR_85_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i], "SpR_85_quality_total_execution")
                            m.addConstr(SpR_70_quality_total_execution[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= SpR_70_quality_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i], "SpR_70_quality_total_execution")
        # optimizing the model
        m.Params.NonConvex = 2
        m.Params.MIPGap = 0.01
        m.Params.NodeLimit = 1000000
        m.Params.TimeLimit = 5000.0
        m.optimize()

        # os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # m.computeIIS()
        # m.write("model1.ilp")

        print("-----------------------------")
        # 儲存optimize
        self.pcs_cap = pcs_cap.x*Cap_ESS_scale
        self.battery_cap = battery_cap.x*Cap_ESS_scale
        self.ESS_initail_cost = ESS_initial_cost.x
        self.EssSOC_c = EssSOC_c.x
        self.EssSOC_dsc = EssSOC_dsc.x
        # self.Essdeg_c = Essdeg_c.x
        self.Essdeg_dsc = Essdeg_dsc.x
        self.Contr_cap = Contr_cap.x*1000 * Pcap_scale 
        self.Ccap = Contr_cap.x*1000 * Pcap_scale * self.Ccap_base
        # SpR
        if(ASS_status == 1):
            for i in len_n:
                self.Cap_SpR = np.append(self.Cap_SpR, Cap_SpR[int(i/4)].x * Cap_SpR_scale)
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums): 
                        self.SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0] = SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0].x
                        self.sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] = SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0].x
                        for i in len_n:
                            # SpR energy price
                            self.SpR_energy_execution_record[TOU_scenario][ASS_time_scenario][netload_scenario][i] = SpR_energy_execution[TOU_scenario][ASS_time_scenario][netload_scenario][i].x
        # netload
        for netload_scenario in range(self.netload_scenario_nums): 
            for i in len_n:
                self.netload[netload_scenario][i] = self.netload_scenario_power.iloc[netload_scenario,i]
        # Another data
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums): 
                    for i in len_n:
                        # ESS 
                        self.PESS[TOU_scenario][ASS_time_scenario][netload_scenario][i] = Pess_c[TOU_scenario][ASS_time_scenario][netload_scenario][i].x-Pess_dsc[TOU_scenario][ASS_time_scenario][netload_scenario][i].x
                        self.ESS_SOC[TOU_scenario][ASS_time_scenario][netload_scenario][i] = SOC_ESS[TOU_scenario][ASS_time_scenario][netload_scenario][i].x
                        # grid
                        self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] = Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i].x
        print("-----------------------------")
        print("Pcap",Contr_cap.x*1000 * Pcap_scale)
        print("battery_cap",battery_cap.x*Cap_ESS_scale)
        print("pcs_cap",pcs_cap.x*Cap_ESS_scale)
        print("ESS_initail_cost",ESS_initial_cost.x)
        if(ASS_status == 1):
            print("Cap_SpR",np.mean(self.Cap_SpR.reshape(-1, 4)[:24], axis=1))
        # print("total_Ch_deg_cost", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Essdeg_c.x * Pess_c[TOU_scenario][ASS_day_scenario][netload_scenario][i].x for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))) )
        # print("total_Dsc_deg_cost", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Essdeg_dsc.x * Pess_dsc[TOU_scenario][ASS_day_scenario][netload_scenario][i].x for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))) )
        print("total_Ccap_cost", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * Contr_cap.x*1000 * Pcap_scale * self.Ccap_base for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))))
        print("total_Ccap1_cost", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * MAX_Ccap1[TOU_scenario][ASS_day_scenario][netload_scenario][0].x for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))))
        print("total_Ccap2_cost", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * MAX_Ccap2[TOU_scenario][ASS_day_scenario][netload_scenario][0].x for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))))
        print("SpR_fail_times", gp.quicksum(self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_day_scenario] * self.netload_scenario_days[netload_scenario] * U_SpR_fail[TOU_scenario][ASS_day_scenario][netload_scenario][0].x for i in len_n for netload_scenario in range(self.netload_scenario_nums)for ASS_day_scenario in range(len(self.ASS_time))for TOU_scenario in range(len(self.tou_uncer_prob))))
    def optimal_result_ouput(self):
        # 淨負載欄位資料
        self.Pbuild = np.array([[0.0 for i in self.len_n] for netload_scenario in range(self.netload_scenario_nums)])
        for netload_scenario in range(self.netload_scenario_nums): 
            for i in self.len_n:
                self.Pbuild[netload_scenario][i] = self.netload[netload_scenario][i]
        # Ccap1欄位資料
        self.Ccap1 = np.array([[[[0.0 for i in self.len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.Ccap1_max = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])   
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums): 
                    for i in self.len_n:
                        if self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= self.Contr_cap:
                            self.Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][i] = (self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] - self.Contr_cap) * self.Ccap_base * 2
                        else:
                            self.Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario][i] = 0
                    self.Ccap1_max[TOU_scenario][ASS_time_scenario][netload_scenario][0] = max(self.Ccap1[TOU_scenario][ASS_time_scenario][netload_scenario])   
        # Ccap2欄位資料
        self.Ccap2 = np.array([[[[0.0 for i in self.len_n] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        self.Ccap2_max = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])   
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums): 
                    for i in self.len_n:
                        if self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] >= self.Contr_cap * 1.1:
                            self.Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][i] = (self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] - self.Contr_cap * 1.1) * self.Ccap_base
                        else:
                            self.Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario][i] = 0
                    self.Ccap2_max[TOU_scenario][ASS_time_scenario][netload_scenario][0] = max(self.Ccap2[TOU_scenario][ASS_time_scenario][netload_scenario])   
        # Pnet
        Pnet_cost_allday = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums): 
                    for i in self.len_n:
                        Pnet_cost_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += (self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][i] * self.TOU_scenario[TOU_scenario][i])
        # 原始淨負載用電成本
        Pbuild_cost = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
        for TOU_scenario in range(len(self.tou_uncer_prob)):
            for ASS_time_scenario in range(len(self.ASS_time)):
                for netload_scenario in range(self.netload_scenario_nums):
                            Pbuild_cost[TOU_scenario][ASS_time_scenario][netload_scenario][0] = sum(self.Pbuild[netload_scenario] * self.TOU_scenario[TOU_scenario])
        if(self.spr_status == 1):
            # sbspm
            SpR_sbspm = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0: 
                            SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] = round((self.SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0]-(sum(self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario][(self.ASS_time[ASS_time_scenario]):(self.ASS_time[ASS_time_scenario]+4)])/4))/(self.Cap_SpR[self.ASS_time[ASS_time_scenario]]*1000), 3)
                        else:
                            SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] = 1
            # SpR容量費與效能費
            SpR_cap_eff_price_allday = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        if ASS_time_scenario > 0:
                            for i in self.len_n:
                                if self.ASS_time[ASS_time_scenario] <= i < self.ASS_time[ASS_time_scenario] + 4:
                                    if SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] >= 0.95:
                                        SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += (self.SpR_eff_price[0] + self.SpR_cap_price[0])*1*self.Cap_SpR[int(self.ASS_time[ASS_time_scenario]/4)]
                                    elif 0.85 <= SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] < 0.95:
                                        SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += (self.SpR_eff_price[0] + self.SpR_cap_price[0])*0.7*self.Cap_SpR[int(self.ASS_time[ASS_time_scenario]/4)]
                                    elif 0.7 <= SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0] < 0.85:
                                        SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += 0
                                    else:
                                        SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += (self.SpR_eff_price[0] + self.SpR_cap_price[0])*(-240)*self.Cap_SpR[int(self.ASS_time[ASS_time_scenario]/4)]
                                else:
                                    SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] += (self.SpR_eff_price[0] + self.SpR_cap_price[0])*self.Cap_SpR[int(i/4)]
                        else:
                            SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0] = sum((self.SpR_eff_price[0] + self.SpR_cap_price[0])*4*self.Cap_SpR[i] for i in range(self.time_interval))
            # 即時備轉電能費收益
            SpR_energy_exe_rev = np.array([[[[0.0] for netload_scenario in range(self.netload_scenario_nums)]for ASS_day_scenario in range(len(self.ASS_time))]for TOU_scenario in range(len(self.tou_uncer_prob))])
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        SpR_energy_exe_rev[TOU_scenario][ASS_time_scenario][netload_scenario][0] = sum(self.SpR_energy_execution_record[TOU_scenario][ASS_time_scenario][netload_scenario])
        # -----------------------------------儲存結果-----------------------------------
        # 建立儲存結果資料夾
        self.folder_path = 'D:/EPSLab_pc/Thomas_碩論/thesis/output_MuiltSpR'+str(self.mins)+'/Contract_'+str(self.Contr_cap)
        try:
            path = Path(self.folder_path)
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"建立資料夾失敗：{error}")
        os.chdir(self.folder_path)
        
        # 輸出最佳規劃容量
        optimal_capacity = {'battery_cap':[self.battery_cap], 'pcs_cap':[self.pcs_cap], 'contr_cap':[self.Contr_cap]}
        pd.DataFrame(optimal_capacity).to_csv('optimal_capacity.csv', encoding='utf-8_sig', index=False)

        if(self.spr_status == 1):
            # 輸出各代表日最佳化排程結果
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        # 輸出csv
                        data_ = {'Pbuild': self.Pbuild[netload_scenario],
                                'PESS': self.PESS[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'Pnet': self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'contract limit': np.array([self.Contr_cap] * len(self.time)),
                                'CTOU': self.TOU_scenario[TOU_scenario],
                                'ESS SOC': self.ESS_SOC[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'Ccap1': np.array([self.Ccap1_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]] * len(self.time)),
                                'Ccap2': np.array([self.Ccap2_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]] * len(self.time)),
                                'Cap_SpR': self.Cap_SpR,
                                'SpR energy price': self.SpR_energy_price,
                                'SpR energy exection': self.SpR_energy_execution_record[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'P SpR CBL': np.array([self.SpR_CBL_base[TOU_scenario][ASS_time_scenario][netload_scenario][0]] * len(self.time))}
                        df = pd.DataFrame(data_)
                        df.to_csv('day_optimization_' + str(TOU_scenario) + '_' + str(ASS_time_scenario) + '_' + str(netload_scenario) + '.csv', sep=',', index=False)
            # 輸出投標量
            cap_spr_data = {'Cap_SpR':np.mean(self.Cap_SpR.reshape(-1, 4)[:24], axis=1)}
            pd.DataFrame(cap_spr_data).to_csv('Cap_SpR_optimization.csv', encoding='utf-8_sig', index=False)
            # 輸出總成本
            cnt = 1
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        # 輸出csv
                        data = {'Case':[cnt],
                                'days':[self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_time_scenario] * self.netload_scenario_days[netload_scenario]],
                                'Pbuild_cost':[Pbuild_cost[TOU_scenario][ASS_time_scenario][netload_scenario][0]],   
                                'Ccap':[self.Ccap],
                                'Ccap1_max':[self.Ccap1_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'Ccap2_max':[self.Ccap2_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'Pnet_cost':[Pnet_cost_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'SpR_sbspm':[SpR_sbspm[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'SpR_cap_eff_price':[SpR_cap_eff_price_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'SpR_energy_exe_rev':[SpR_energy_exe_rev[TOU_scenario][ASS_time_scenario][netload_scenario][0]]}
                        self.optimize_cost_data = pd.concat([self.optimize_cost_data, pd.DataFrame(data)], ignore_index=True)
                        cnt += 1
            self.optimize_cost_data.to_csv('2021_ESS_build_optimize_cost_data.csv', encoding='utf-8_sig', index=False)
        else:
            # 輸出各代表日最佳化排程結果
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        # 輸出csv
                        data_ = {'Pbuild': self.Pbuild[netload_scenario],
                                'PESS': self.PESS[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'Pnet': self.Pnet[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'contract limit': np.array([self.Contr_cap] * len(self.time)),
                                'CTOU': self.TOU_scenario[TOU_scenario],
                                'ESS SOC': self.ESS_SOC[TOU_scenario][ASS_time_scenario][netload_scenario],
                                'Ccap1': np.array([self.Ccap1_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]] * len(self.time)),
                                'Ccap2': np.array([self.Ccap2_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]] * len(self.time)),
                                'Cap_SpR': np.array([0] * len(self.time)),
                                'SpR energy price': np.array([0] * len(self.time)),
                                'SpR energy exection': np.array([0] * len(self.time)),
                                'P SpR CBL': np.array([0] * len(self.time))}
                        df = pd.DataFrame(data_)
                        df.to_csv('day_optimization_' + str(TOU_scenario) + '_' + str(ASS_time_scenario) + '_' + str(netload_scenario) + '.csv', sep=',', index=False)
            # 輸出投標量
            cap_spr_data = {'Cap_SpR':np.array([0] * self.time_interval)}
            pd.DataFrame(cap_spr_data).to_csv('Cap_SpR_optimization.csv', encoding='utf-8_sig', index=False)
            # 輸出總成本
            cnt = 1
            for TOU_scenario in range(len(self.tou_uncer_prob)):
                for ASS_time_scenario in range(len(self.ASS_time)):
                    for netload_scenario in range(self.netload_scenario_nums):
                        # 輸出csv
                        data = {'Case':[cnt],
                                'days':[self.tou_uncer_prob[TOU_scenario] * self.ASS_time_uncer_prob[ASS_time_scenario] * self.netload_scenario_days[netload_scenario]],
                                'Pbuild_cost':[Pbuild_cost[TOU_scenario][ASS_time_scenario][netload_scenario][0]],   
                                'Ccap':[self.Ccap],
                                'Ccap1_max':[self.Ccap1_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'Ccap2_max':[self.Ccap2_max[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'Pnet_cost':[Pnet_cost_allday[TOU_scenario][ASS_time_scenario][netload_scenario][0]],
                                'SpR_sbspm':[0],
                                'SpR_cap_eff_price':[0],
                                'SpR_energy_exe_rev':[0]}
                        self.optimize_cost_data = pd.concat([self.optimize_cost_data, pd.DataFrame(data)], ignore_index=True)
                        cnt += 1
            self.optimize_cost_data.to_csv('2021_ESS_build_optimize_cost_data.csv', encoding='utf-8_sig', index=False)

        # 成本分析
        org_cost = sum(self.optimize_cost_data['days'] * self.optimize_cost_data['Pbuild_cost'])
        org_cap = 365*self.planning_area_cc*1000*self.Ccap_base
        net_cost = sum(self.optimize_cost_data['days'] * self.optimize_cost_data['Pnet_cost'])
        spr_rev = sum(self.optimize_cost_data['days'] * (self.optimize_cost_data['SpR_cap_eff_price']+self.optimize_cost_data['SpR_energy_exe_rev']))
        optimal_cap_cost = sum(self.optimize_cost_data['days'] * self.optimize_cost_data['Ccap'])
        over_cap_cost = sum(self.optimize_cost_data['days'] * (self.optimize_cost_data['Ccap1_max']+self.optimize_cost_data['Ccap2_max']))
        
        ess_replace_cost = 0
        r,Y = self.discount_rate/100, self.planning_year
        if(self.planning_year > self.ess_calender_life):
            for y in range(1, Y):
                if(y != Y and y%self.ess_calender_life == 0):
                    ess_replace_cost += (1/(1+r)**(y))*self.ESS_initail_cost
        print("ess_replace_cost : ",ess_replace_cost)
        # ess_annual_cost = (self.ESS_initail_cost + ess_replace_cost)/self.planning_year
        ess_annual_cost = (self.ESS_initail_cost + ess_replace_cost)*(r*((1+r)**Y)/(((1+r)**Y)-1))
        print("ess_annual_cost",ess_annual_cost)
        org_microgrid_cost = org_cost+org_cap
        optimal_microgrid_cost = ess_annual_cost+net_cost+optimal_cap_cost+over_cap_cost-spr_rev

        self.cost_analysis = {
            'case':['微電網原始用電成本','加入儲能與電能管理系統之微電網用電成本'],
            'ESS_initail_cost':[0,self.ESS_initail_cost],
            'ess_replace_cost':[0,ess_replace_cost],
            'annual_ess_build_cost':[0,ess_annual_cost],
            'grid':[org_cost,net_cost],
            'cc_cost':[org_cap,optimal_cap_cost],
            'over_cc':[0,over_cap_cost],
            'spr_rev':[0,spr_rev],
            'total_cost':[org_microgrid_cost,optimal_microgrid_cost],
            'total_save':[0,org_microgrid_cost-optimal_microgrid_cost]
        }
        pd.DataFrame(self.cost_analysis).to_csv('cost_analysis.csv', encoding='utf-8_sig', index=False)

if __name__ == '__main__':  
    # =======================================================================
    # 即時最佳化函式
    opt_realtime = OnGrid_realtime_opt(15)
    # 最佳化參數
    opt_realtime.realtime_opt_parameter() 
    # -----------最佳化-----------
    opt_realtime.centralize_optimization()
    # ---------輸出排程結果---------
    opt_realtime.optimal_result_ouput()

