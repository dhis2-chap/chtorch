
from chapkit import (
    SqlAlchemyChapDatabase,
)
from chapkit.model import (
    ChapModelService,
)
from sklearn.linear_model import LinearRegression
from chtorch.runner import MyRunner
from chtorch.runner import MyConfig
from chtorch.runner import info


database = SqlAlchemyChapDatabase("target/chapkit.db", config_type=MyConfig)
runner = MyRunner(info, database, config_type=MyConfig)

app = ChapModelService(
    runner=runner,
    database=database,
).create_fastapi()
