
"""
deploiement_prefect.py
Création des déploiements Prefect avec une exécution quotidienne.
"""

from datetime import timedelta
from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow


if __name__ == "__main__":

    # Déploiement du flow ALL
    deployment_all = all_flow.to_deployment(
        name="ml-pipeline-all",
        interval=timedelta(days=1)
    )

    # Déploiement du flow TRAIN
    deployment_train = train_flow.to_deployment(
        name="ml-pipeline-train",
        interval=timedelta(days=1)
    )

    # Déploiement du flow EVALUATE
    deployment_evaluate = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        interval=timedelta(days=1)
    )

    # Enregistrement et lancement des déploiements
    serve(
        deployment_all,
        deployment_train,
        deployment_evaluate
    )
