import uuid
import pandas as pd
import yaml
from datetime import datetime
from pulsar_data_collection.data_capture import DataCapture, DatabaseLogin
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.metrics.base import CustomMetric 
from roi import ReturnOfInvestment
from sklearn.metrics import confusion_matrix



# Charger constantes depuis le yaml
with open('src/constants.yaml', 'r') as file:
    config = yaml.safe_load(file)

@CustomMetric
def test_roi(current, reference, **kwargs):
    roi_instance = ReturnOfInvestment()
    roi_x = roi_instance.calculate_ROI(
        reference,
        current,
        config['roi']['fpr_threshold'],
        config['roi']['fnr_threshold'],
        config['roi']['cost_per_fn'],     
        config['roi']['cost_per_fp']
    )
    return roi_x

#Calcule le cout total des mauvaises prédictions (FP + FN)
@CustomMetric
def cost_wrong_prediction(current, reference, **kwargs):
    cm = confusion_matrix(reference, current)
    f_p = cm[0,1]
    f_n = cm[1,0]
    cost_fp = f_p * config['roi']['cost_per_fp']
    cost_fn = f_n * config['roi']['cost_per_fn']
    total_cost = cost_fp + cost_fn
    return total_cost



#charger donnees de ref qui sont dans le csv
if __name__ == "__main__":

    df_result = pd.DataFrame()
    df_ref = pd.read_csv(config['data']['reference_path'])
    df_ref['pred_timestamp'] = datetime.strptime('10/7/2022 22:30:30', '%d/%m/%Y %H:%M:%S')
    df_ref['period'] = 'reference'
    df_ref['model_id'] = config['model']['id']
    df_ref['model_version'] = config['model']['version']



#login influxdb
    database_login = DatabaseLogin(**config['database'])

    dat_capture = DataCapture(
        storage_engine="influxdb",
        operation_type="METRICS",
        login_url=database_login
    )

    #recuperer nouvelles predictions
    last_eval_timestamp = dat_capture.collect_eval_timestamp()
    if last_eval_timestamp:
        last_eval_timestamp_str = last_eval_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        db_df = pd.DataFrame(dat_capture.collect({"time": f">= '{last_eval_timestamp_str}'"}).get("prediction"))
    else:
        db_df = pd.DataFrame(dat_capture.collect().get("prediction"))

    print(f"Dataframe collected, df length: {len(db_df)}")

    if len(db_df) == 0:
        print("Aucune nouvelle prédiction trouvée")
        exit()

    #conversion yes-no et conversion des donnes de ref
    df_ref['default'] = df_ref['default'].map(config['data']['conversion_mapping'])
    db_df['pred_timestamp'] = datetime.utcnow()

    #calcule drift metrics and performance metrics
    analysis = Analyzer(name='Compute Script Analyzer', description='Analyzer for compute script usage', model_id = config['model']['id'], model_version = config['model']['version'])
    analysis.add_drift_metrics(
        metrics_list= config['drift_metrics'],
        features_list= config['features'] 
    )
    analysis.add_performance_metrics(metrics_list=config['performance_metrics'], y_name = config['model']['target_column'])

    #calcule et stocke dans analysis toutes les metrics drift et perf
    analysis.run(reference=df_ref, current=db_df, options=config['metric_options'])
        
    #transforme les resultat en dataframe
    df_result_drift = analysis.results_to_pandas()

    #extraire vrai valeur et prediction pour le calcule roi
    reference_data = db_df[config['model']['target_column']].astype(int).tolist()
    current_data = db_df[config['model']['prediction_column']].tolist()

    #calcule roi (df_roi)
    cus = test_roi(metric_name = 'roi')
    cus.evaluate(current = current_data, reference = reference_data)
    df_roi = cus.get_result()

    #roi en dataframe
    df_roi_pd = pd.DataFrame(df_roi)
        
    #ajouter les metadonnees
    df_reshaped = df_roi_pd.set_index(0).T
    df_reshaped["period_start"] = db_df.pred_timestamp.min()
    df_reshaped["period_end"] = db_df.pred_timestamp.max()
    df_reshaped["eval_timestamp"] = datetime.utcnow()
    df_reshaped["model_id"] = config['model']['id']
    df_reshaped["model_version"] = config['model']['version']

    #cost wrong prediction
    cus_cost = cost_wrong_prediction(metric_name = 'cost_wrong_prediction')
    cus_cost.evaluate(current = current_data, reference = reference_data)
    df_cost = cus_cost.get_result()

    df_cost_pd = pd.DataFrame(df_cost)
    #metadonnes
    df_cost_reshaped = df_cost_pd.set_index(0).T
    df_cost_reshaped["period_start"] = db_df.pred_timestamp.min()
    df_cost_reshaped["period_end"] = db_df.pred_timestamp.max()
    df_cost_reshaped["eval_timestamp"] = datetime.utcnow()
    df_cost_reshaped["model_id"] = config['model']['id']
    df_cost_reshaped["model_version"] = config['model']['version']
    
     
    #combiner toutes les metrics en 1 dataframe
    df_result_drift = df_result_drift.append(df_reshaped, ignore_index=True)
    df_result_drift = df_result_drift.append(df_cost_reshaped, ignore_index=True) 

    #formater le tout pour influxdb
    df_result_drift.set_index("eval_timestamp", inplace=True)
    df_result_drift["uuid"] = [uuid.uuid4() for _ in range(len(df_result_drift.index))]
    df_result_drift["period_start"] = df_result_drift["period_start"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    df_result_drift["period_end"] = df_result_drift["period_end"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    #push le tout dans influxdb
    dat_capture.push_metrics(df_result_drift)
    print(f"Metrics pushed to the db")

    #enregistre derneire execution date-heure
    eval_timestamp_df = pd.DataFrame({"uuid": uuid.uuid4(), "timestamp": datetime.utcnow(),
                                          "eval_timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}, index=[0, ])
    eval_timestamp_df.set_index("timestamp", inplace=True)
    dat_capture.push_eval_timestamp(eval_timestamp_df)
    print(f"Eval timestamp is updated in the db. Timestamp is: {eval_timestamp_df}")




