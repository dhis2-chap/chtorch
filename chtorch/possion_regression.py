import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def regress_dataframe(data):
    # data = pd.DataFrame(
    #     {
    #         'disease_cases': [1, 2, 3, 4, 5, 6],
    #         'location': ['A', 'A', 'B', 'B', 'C', 'C'],
    #         'season': list(range(1, 7)),
    #     }
    # )
    model = smf.glm(
        formula="disease_cases ~ C(location) + C(season) + C(location):C(season)",
        data=data,
        family=sm.families.Poisson()).fit()

    return model


def regress(dataset: DataSet):
    df = dataset.to_pandas()
    df = df[df['disease_cases'] > 0]
    df['season'] = df['time_period'].dt.month
    model = regress_dataframe(df)
    print(model.predict(df, linear=True))
    df['log_predicted'] = model.predict(df, linear=True)
    df['log_cases'] = np.log(df['disease_cases'])
    df['residuals'] = df['log_predicted']- df['log_cases']
    px.scatter(df, x='log_predicted', y='log_cases', color='location').show()
    px.violin(df, x='season', y='residuals', color='location').show()
    px.histogram(df, x='residuals', color='location').show()
    return model
    # print(df['time_period'])


if __name__ == '__main__':
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/full_data/vietnam.csv', FullData)

    model = regress(dataset)
