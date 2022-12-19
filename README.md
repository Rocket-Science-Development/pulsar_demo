<h2>pulsar_demo</h2>

Pulsar Data Collection is environment what shows usage of `pulsar-metrics
`, `pulsar-data-collection` SDKs with artificial data and shows results in real-time
on dashboards.

<h3>Getting started</h3><hr/>

<h4>Components:</h4>
There are several components in pulsar-demo: grafana dashboards,
model, script for metrics computing, script for making fake data (you can push own data instead),
influx database image, docker-compose file what combines everything together, crontab bash script
what computes metrics each X minutes/hours/days/etc.

Graphana dashboards - this component has already predefined dashboards 
and datasources, what allows to see results without additional set up.

Metrics computing script - uses `pulsar-data-collection` SDK for retrieving
data from the database and pushing calculated metrics back, and `pulsar-metrics` SDK for metrics calculation.

Influx database docker build - dockerfile with 1.5.3 version of InfluxDB and entrypoint file what creates user on app start.

Crontab script - bash script what creates crontab inside docker.

Docker-compose - combines all apps together.


#### How to run locally:
`docker-compose up`


TODO: Add screenshot of Graphana





