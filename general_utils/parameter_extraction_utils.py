import numpy as np
import pandas as pd

def get_portfolio_parameters(Prices, ESG_df, index_Prices):

    returns = np.diff(Prices, axis=0)/Prices[:-1,:]

    expected_returns = np.nanmean(returns,axis=0)

    df_returns = pd.DataFrame(returns)
    covariance_matrix = df_returns.cov().to_numpy()

    n = len(expected_returns)
    coskewness_tensor = np.zeros((n,n,n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                coskewness_tensor[i,j,k] = (np.nanmean((returns[:,i] - expected_returns[i])*(returns[:,j] - expected_returns[j])*(returns[:,k] - expected_returns[k])))/(np.sqrt(covariance_matrix[i,i])*np.sqrt(covariance_matrix[j,j])*np.sqrt(covariance_matrix[k,k]))

    ESG_scores = ESG_df.apply(lambda col: col[col.last_valid_index()] if col.last_valid_index() is not None else None).to_numpy()

    if index_Prices is not None:

        index_returns=np.diff(index_Prices, axis=0)/index_Prices[:-1]

        df_index_returns=pd.DataFrame(index_returns)

        market_var=df_index_returns.var()
        market_var=float(market_var.iloc[0])
        market_cov=[]
        for i in range(n):
            X=pd.concat([df_returns.iloc[:,i], df_index_returns], axis=1).cov()
            market_cov.append(X.iloc[0,1])

        market_cov=np.array(market_cov)
        betas=market_cov/market_var
    
    else:
        betas = None

    return expected_returns, covariance_matrix, coskewness_tensor, ESG_scores, betas






