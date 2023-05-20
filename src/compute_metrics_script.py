import uuid
import logging
import pandas as pd
from datetime import datetime
from pulsar_data_collection.data_capture import DataCapture, DatabaseLogin, DataWithPrediction
from pulsar_metrics.analyzers.base import Analyzer


if __name__ == "__main__":
    # Reading reference dataset
    df_result = pd.DataFrame()
    df_ref = pd.read_csv('/app/datasets/pokemon.csv')

    df_ref['model_id'] = '1'
    df_ref['model_version'] = '2'
    df_ref['pred_timestamp'] = datetime.strptime('10/7/2022 22:30:30', '%d/%m/%Y %H:%M:%S')
    df_ref['period'] = 'reference'

    # Reading the newest prediction data
    database_login = DatabaseLogin(db_host="influxdb", db_port=8086, db_user="admin", db_password="pass123", protocol="line")

    dat_capture = DataCapture(
        storage_engine="influxdb",
        operation_type="METRICS",
        login_url=database_login
    )

    # receiving the last period
    last_eval_timestamp = dat_capture.collect_eval_timestamp()
    if last_eval_timestamp:
        last_eval_timestamp_str = last_eval_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        db_df = pd.DataFrame(dat_capture.collect({"time": f">= '{last_eval_timestamp_str}'"}).get("prediction"))
    else:
        db_df = pd.DataFrame(dat_capture.collect().get("prediction"))

    print(f"Dataframe collected, df length: {len(db_df)}")

    if len(db_df):
        # TODO: it should be changed, it's not clear why we cannot use datetime now
        db_df['pred_timestamp'] = datetime.strptime('20/8/2022 22:30:30', '%d/%m/%Y %H:%M:%S')

        analysis = Analyzer(name='Compute Script Analyzer', description='Analyzer for compute script usage', model_id='1', model_version='2')
        analysis.add_drift_metrics(
            metrics_list=['wasserstein', 'ttest', 'ks_2samp','kl','manwu','levene','bftest','CvM','psi'],
            # features_list=['age','al','ane','appet','ba','bgr','bp','bu','cad','dm','hemo','htn','id','pc','pcc','pcv','pe','pot','rbc','rbcc','sg','sc','sod','su']
            features_list=['#','Attack','Defense','Generation','HP','Legendary','Sp. Atk','Sp. Def','Speed','Total']
        )

        analysis.run(reference=df_ref, current=db_df, options={'ttest': {'alpha': 0.05, 'equal_var': False}, 'wasserstein': {'threshold' : 0.2}})
        df_result_drift = analysis.results_to_pandas()

        df_result_drift.set_index("eval_timestamp", inplace=True)

        df_result_drift["uuid"] = [uuid.uuid4() for _ in range(len(df_result_drift.index))]
        df_result_drift["period_start"] = df_result_drift["period_start"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df_result_drift["period_end"] = df_result_drift["period_end"].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        dat_capture.push_metrics(df_result_drift)

        print(f"Metrics pushed to the db")

        # Add the last period to db after pushing
        eval_timestamp_df = pd.DataFrame({"uuid": uuid.uuid4(), "timestamp": datetime.utcnow(),
                                          "eval_timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}, index=[0, ])
        eval_timestamp_df.set_index("timestamp", inplace=True)
        dat_capture.push_eval_timestamp(eval_timestamp_df)
        print(f"Eval timestamp is updated in the db. Timestamp is: {eval_timestamp_df}")

