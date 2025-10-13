

from chapkit import ChapServiceInfo




info = ChapServiceInfo(
    author="Knut Rand",
    author_note="This model might need configuration of hyperparameters in order to work properly. When the model shows signs of overfitting, reduce 'state_dim' and/or increase 'dropout' and 'weight_decay'.",
    author_assessed_status="red",
    contact_email="knutdrand@gmail.com",
    description="This is a deep learning model template for CHAP. It is based on pytorch and can be used to train and predict using deep learning models. This typically need some configuration to fit the specifics of a dataset.",
    display_name="Torch Deep Learning Model",
    organization="HISP Centre, University of Oslo",
    organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
    citation_info='Climate Health Analytics Platform. 2025. "Torch Deep Learning Model". HISP Centre, University of Oslo. https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html',
    #data_configuration = {
    #    "required_covariates": ["population"],
    #    "target": "disease_cases",
    #    "allow_free_additional_continuous_covariates": True,
    #    "supported_period_type": "any",
    #}
)
