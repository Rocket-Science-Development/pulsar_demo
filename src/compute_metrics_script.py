import uuid
import pandas as pd
from datetime import datetime
from pulsar_data_collection.data_capture import DataCapture, DatabaseLogin
from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.metrics.base import CustomMetric 
from roi import ReturnOfInvestment

FPR_THRESHOLD=0.5
FNR_THRESHOLD=0.2
cost_per_FP=100
cost_per_FN=300

@CustomMetric
def test_roi(current, reference, multiple=3, **kwargs):
    roiInstance = ReturnOfInvestment()
    roi_x = roiInstance.calculate_ROI(reference, current, FPR_THRESHOLD, FNR_THRESHOLD, cost_per_FP, cost_per_FN)
    return roi_x

if __name__ == "__main__":
    # Reading reference dataset
    df_result = pd.DataFrame()
    df_ref = pd.read_csv('/app/datasets/bank_num.csv')
    target = 'default'
    prediction = 'y_pred'

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
    print(f"Dataframe : {db_df.head()}")

    conversion_dict = {'yes': 1, 'no': 0}
    df_ref['default'] = df_ref['default'].map(conversion_dict)

    if len(db_df):
        # TODO: it should be changed, it's not clear why we cannot use datetime now
        db_df['pred_timestamp'] = datetime.strptime('20/8/2022 22:30:30', '%d/%m/%Y %H:%M:%S')

        analysis = Analyzer(name='Compute Script Analyzer', description='Analyzer for compute script usage', model_id='1', model_version='2')
        analysis.add_drift_metrics(
            metrics_list=['wasserstein', 'ttest', 'ks_2samp','kl','manwu','levene','bftest','CvM','psi'],
            # features_list=['age','al','ane','appet','ba','bgr','bp','bu','cad','dm','hemo','htn','id','pc','pcc','pcv','pe','pot','rbc','rbcc','sg','sc','sod','su']
            # features_list=['#','Attack','Defense','Generation','HP','Sp. Atk','Sp. Def','Speed','Total']
            features_list=['age','job','marital','education','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','deposit']
        )
        analysis.add_performance_metrics(metrics_list=['accuracy','precision','recall'], y_name = target)

        analysis.run(reference=df_ref, current=db_df, options={'ttest': {'alpha': 0.05, 'equal_var': False}, 'wasserstein': {'threshold' : 0.2}})
        print("analysis: ", analysis)
        df_result_drift = analysis.results_to_pandas()

        print("df_result",df_result_drift)

        # print("prediction:",db_df['y_pred'])
        # print("ground truth:",db_df['Legendary'])
        # print("Lists of Legendary: ", db_df['Legendary'].astype(int).to_list())
        # print("Lists of Prediction: ", db_df['y_pred'].to_list())
        
        # roiInstance = ReturnOfInvestment()
        # roi_x = roiInstance.calculate_ROI(db_df['Legendary'].astype(int).to_list(), db_df['y_pred'].to_list())
        # print("roi return: ", roi_x)

        # conversion_dict = {'yes': 1, 'no': 0}
        # reference_data = db_df[target].map(conversion_dict).tolist()
        reference_data = db_df[target].astype(int).tolist()
        current_data = db_df[prediction].tolist()

        cus = test_roi(metric_name = 'roi')
        cus.evaluate(current = current_data, reference = reference_data)
        df_roi = cus.get_result()
        print("df_roi: ", df_roi)

        df_roi_pd = pd.DataFrame(df_roi)
        print("df_roi_pd: ", df_roi_pd)

        # df_reshaped = pd.DataFrame([df_roi_pd[1].to_list()], columns=df_roi_pd[0])
        # df_reshaped = df_roi_pd.pivot(index=None, columns=df_roi_pd.columns[0], values=df_roi_pd.columns[1])
        df_reshaped = df_roi_pd.set_index(0).T
        df_reshaped["period_start"] = db_df.pred_timestamp.min()
        df_reshaped["period_end"] = db_df.pred_timestamp.max()
        df_reshaped["eval_timestamp"] = datetime.now()
        df_reshaped["model_id"] = df_ref["model_id"]
        df_reshaped["model_version"] = df_ref["model_version"]

        print("df_reshaped: ", df_reshaped)

        df_result_drift = df_result_drift.append(df_reshaped, ignore_index=True)
        print("df_result_drift: ", df_result_drift)

        
        
        # @CustomMetric
        # def test_roi(current, reference, multiple=3, **kwargs):
        #     return multiple*np.max(current - reference)

        # cus = test_roi(metric_name = 'test')
        # cus.evaluate(current = current_data, reference = reference_data, threshold = 1, multiple=0.5)
        # df_roi = cus.get_result()
        # print("df_roi: ", df_roi)
      

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


        def result_to_pandas(self):
            result = None
            if self._results is not None:
                results = pd.DataFrame.from_records([self._results[i].dict() for i in range(len(self._results))])
                for key, value in self._metadata.items():
                    if key not in ["name", "description"]:
                        results[key] = value

            return result


