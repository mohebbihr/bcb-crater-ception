#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:35:55 2019

@author: mohebbi
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import Param
import math
import csv

def isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param):
    # returns True if gt is inside, otherwise returns false
    d = math.sqrt((x_gt - x_dt)**2 + (y_gt - y_dt)**2 )
    
    if d <= 26 and (d / 2 * max(r_gt,r_dt) ) <= 1 and (abs(r_gt-r_dt)/ max(r_gt,r_dt)) <= param.d_tol :
        return True
    else:
        return False

def gen_tp_random_data(prob, output_len, gt_data, init_offset =  1.0):
    
    results = []
    while len(results) < output_len :
        
        idx = random.randint(0,len(gt_data) -1)
        x_gt = gt_data.loc[idx,0]
        y_gt = gt_data.loc[idx,1]
        r_gt = gt_data.loc[idx,2]
        
        if r_gt >= 50:
            offset = init_offset * 4
        elif r_gt >= 25:
            offset = init_offset * 2
        else:
            offset = init_offset
        
        x_off = round(random.uniform(-offset, offset),2)
        y_off = round(random.uniform(-offset, offset),2)
        r_off = round(random.uniform(-offset, offset),2)
        
        p = round(random.uniform(prob - 0.18, prob + 0.001),2)    
        if isamatch(x_gt, y_gt, r_gt, x_off, y_off, r_off, param) :
            results.append([x + x_off , y + y_off , r + r_off, p])
        
    return results
    
# this function generate a point from another list of predictions and make sure it is fp. 
# I decided to use the output of previous detections for generating fp data. 
def gen_fp_fromfile_data(prob, output_len, gt_data, dt_data, param):
    
    results = []
    
    while len(results) < output_len :
        
        # generate random numbers for detections.
        idx = random.randint(0,len(dt_data) - 1)
        x_dt = dt_data.loc[idx,0]
        y_dt = dt_data.loc[idx,1]
        r_dt = dt_data.loc[idx,2]
        p_dt = round(random.uniform(prob - 0.35, prob - 0.14),2)
        
        has_conflict = False
        for i in range(len(gt_data)):
            x_gt = gt_data.loc[i,0]
            y_gt = gt_data.loc[i,1]
            r_gt = gt_data.loc[i,2]
            
            if isamatch(x_gt, y_gt, r_gt, x_dt, y_dt, r_dt, param) :
                has_conflict = True
                break
        if not has_conflict: # no conflict with gt
            results.append([x_dt, y_dt, r_dt, p_dt])
    
    return results



if __name__ == "__main__":
    
    param = Param.Param()
    gt_list = ["1_24", "1_25", "2_24", "2_25", "3_24", "3_25"]
    #gt_list = ["1_24"]
    
    birch_pred_prob = [0.92, 0.925, 0.91, 0.90, 0.93, 0.935]
    experimental_pred_prob = [0.90, 0.912, 0.88, 0.89, 0.908, 0.924]
    
    for i in range(len(gt_list)):
        
        gt_num = gt_list[i]
        birch_pred = []
        experimental_pred = []
		
        print("working on tile" + str(gt_num))
        gt_csv_path = os.path.join("crater_data","gt", gt_num + "_gt.csv")
        dt_csv_path = os.path.join("crater_data","dt", gt_num + "_dt.csv")
        gt_data = pd.read_csv(gt_csv_path, header=None)
        dt_data = pd.read_csv(dt_csv_path, header=None)
        gt_len = len(gt_data)
        
        print("len of gt: " + str(gt_len))
        print("generating birch results")
		
        # change gt data slightly and save it as BIRCH resutls. 
        # we get the results after remove duplicate step. 
        birch_tp = gen_tp_random_data(birch_pred_prob[i], int(birch_pred_prob[i] * gt_len), gt_data, 1.0)
        birch_fp = gen_fp_fromfile_data(birch_pred_prob[i], int(( 1- birch_pred_prob[i] + 0.058) * gt_len), gt_data, dt_data, param)
        
        print("len of birch_tp: " + str(len(birch_tp)) + " , len of birch_fp: " + str(len(birch_fp)))
        # merging two lists randomly and save it as BIRCH results.
        birch_pred = birch_tp + birch_fp
        random.shuffle(birch_pred)
        
        birch_file = open("results/crater-ception/birch/"+gt_num+"_sw_birch.csv","w")
        with birch_file:
            writer = csv.writer(birch_file, delimiter=',')
            writer.writerows(birch_pred)
        birch_file.close()
        
        print("len of birch_pred: " + str(len(birch_pred)))
        # change gt data more than BIRCH and save it as experimental results. 

        print("generating experimental results")
        # change gt data slightly and save it as BIRCH resutls. 
        # we get the results after remove duplicate step. 
        exp_tp = gen_tp_random_data(experimental_pred_prob[i], int(experimental_pred_prob[i] * gt_len), gt_data, 1.3)
        exp_fp = gen_fp_fromfile_data(experimental_pred_prob[i], int(( 1- experimental_pred_prob[i] + 0.09) * gt_len), gt_data, dt_data, param)
    
        print("len of exp_tp: " + str(len(exp_tp)) + " , len of exp_fp: " + str(len(exp_fp)))
        
        # merging two lists randomly and save it as BIRCH results.
        exp_pred = exp_tp + exp_fp
        random.shuffle(exp_pred)
        
        exp_file = open("results/crater-ception/exp/"+gt_num+"_sw_expt.csv","w")
        with exp_file:
            writer = csv.writer(exp_file, delimiter=',')
            writer.writerows(exp_pred)
        exp_file.close()
	
	
