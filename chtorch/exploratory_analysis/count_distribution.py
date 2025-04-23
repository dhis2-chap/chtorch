import dataclasses

import numpy as np
import plotly.express as px
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclasses.dataclass
class CountDistribution:
    dataset: DataSet

    def plot_distributions(self, transform = lambda x: x):
        for location, data in self.dataset.items():
            disease_cases = data.disease_cases
            x = disease_cases[~np.isnan(disease_cases)]
            x = transform(x)
            mu, std = np.mean(x), np.std(x)
            px.histogram((x-mu)/std, title=f"Distribution of disease cases for {location}").show()


if __name__ == "__main__":
    # Example usage
    #dataset = ISIMIP_dengue_harmonized['vietnam']
    dataset = DataSet.from_csv('~/Data/ch_data/laos_district_and_hospital.csv',
                               FullData)
    count_distribution = CountDistribution(dataset)
    count_distribution.plot_distributions(lambda x: np.log(x+1))
