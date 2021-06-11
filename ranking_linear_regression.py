#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
# %% Import data from component analysis workbook
xlsx=pd.ExcelFile('./Test data/ranking_data_denamed.xlsx')
df_rep=xlsx.parse('reputation data')
df_metrics=xlsx.parse('underlying metrics')
df_rank=xlsx.parse('scores')
# %% function for calculating the upper and lower prediction bands
def cal_pred_bands(x, xd, yd, model, conf=0.95):
    # x = requested points
    # xd = x data (array)
    # yd = y data (array)
    # model = linear regression model (class)

    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(model.coef_) # number of parameters - note: there is contrain on x-intercept 
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - model.predict(xd)) ** 2))
    # Auxiliary definitions
    sx = (x.reshape(1,-1) - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = model.predict(x)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

def pred_model(input_df,x_name,y_name='idf score',center_on='Uni 44',plot='off',cc_col=None):
    # input_df = dataframe containing component scores and metrics for peer universities
    # x_name = column name of indepedent variable in input_df
    # y_name = column name of dependent variable in input_df
    # center_on = name of university to cetner the regression model on

    # reduce input_df to the relevent columns
    df_temp=input_df[[x_name, y_name]].copy()
    if cc_col is not None:
        df_temp['group_color']=input_df[cc_col]
    else:
         df_temp['group_color']='k'
    # Remove infinity points (scored 100)
    df_temp_2=df_temp[~(df_temp[y_name]==np.inf)].reset_index()
    # convert idf score and x-value into arrays with the right format
    yd=df_temp_2[y_name].values
    xd=df_temp_2[x_name].values.reshape(-1,1)
    # generate regression model
    lr_model=LinearRegression().fit(xd, yd)
    # set x-intercept of lr_model to force model through center_on point
    center_row=(input_df['Peer']==center_on) # find row for centering data
    y_center=input_df[center_row][y_name]
    x_center=input_df[center_row][x_name]
    slope=lr_model.coef_ # get slope of lr_model
    x_int=y_center-slope*x_center # calculate new intercept
    lr_model.intercept_=x_int.values[0] # set new intercept for lr_model
    r_sq=lr_model.score(xd, yd)
    print('r-square value = ',r_sq)
    # plot results
    if plot=='on':
        cd=df_temp_2['group_color'] # set color of scatter points
        plot_model(xd, yd, lr_model, cd)
    return lr_model

def plot_model(xd, yd, lr_model,cd):
    # lr_model = linear regression model
    # xd = x data (array)
    # yd = y data (array)
    # cd = colour data (array)

    # set up matlab plot
    fig, lr_plot = plt.subplots()
    # calculate 95% confidence interval for data point
    y_predband=cal_pred_bands(xd, xd, yd, lr_model, conf=0.95)
    # plot points for training data
    for i in range(len(xd)):
        lr_plot.scatter(xd[i], yd[i], color=cd[i])
    # add confidence interval
    lr_plot.plot(xd, y_predband[1].reshape(-1,1),color='g')
    lr_plot.plot(xd, y_predband[0].reshape(-1,1),color='g')
    lr_plot.set_title('QS score prediction')
    lr_plot.set_xlabel('Metric')
    lr_plot.set_ylabel('Predicted idf score')
    return lr_plot

# %% Prepare data for analysing peer universities.
df_peer=df_rank[~df_rank['Peer'].isna()]
df_metrics['Release year']=df_metrics['Rankings Year']-1 # release year =  ranking year - 1
df_metrics_2=pd.pivot_table(df_metrics, 
    index=['Peer','Release year'], columns='Measure Names',values='Measure Values')
df_peer_2=df_peer.merge(df_metrics_2, how='left',left_on=['Peer','Release year'],right_index=True)

# %% create universities groups - Go8, UNSW, others
# optional, only used for coloring scatter plot of regression model
go8_map={'Uni 31':'G1','Uni 40':'G1','Uni 41':'G1',
    'Uni 44':'G2','Uni 46':'G2','Uni 55':'G2','Uni 92':'G2','Uni 106':'G2'}
df_peer_2['uni_group']=df_peer_2['Peer'].map(go8_map).fillna('other')
colors={'G1':'r','G2':'b','other':'k'}
df_peer_2['uni_group_color']=df_peer_2['uni_group'].map(colors)
df_peer_2.reset_index(inplace=True)
# %% Calculate the inverse distribution function (idf); i.e. inverse of CDF
df_peer_3=df_peer_2.copy()
df_peer_3['idf isr score']=df_peer_2['isr score'].apply(lambda x: stats.norm.ppf(x/100))

# %% Make predictive model
isr_model=pred_model(df_peer_3,x_name='International Students per 100 Students - Ratio', center_on='Uni 44',
    y_name='idf isr score',plot='on',cc_col='uni_group_color')

# %%


# %%
