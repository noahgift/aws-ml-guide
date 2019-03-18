---
redirect_from:
  - "/lessons/lesson7-awsml-case-studies"
interact_link: content/lessons/Lesson7_AWSML_Case_Studies.ipynb
kernel_name: python3
title: 'Lesson 7 - Amazon Machine Learning Case Studies'
prev_page:
  url: /lessons/Lesson6_AWSML_Production_ML_App
  title: 'Lesson 6 - Amazon Machine Learning Production Machine Learning Applications'
next_page:
  url: 
  title: ''
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

<a href="https://colab.research.google.com/github/noahgift/aws-ml-guide/blob/master/Lesson7_AWSML_Case_Studies.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lesson 7: Case Studies

[Watch Lesson 7:  Case Studies Video](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597/9780135556597-ACML_01_07_00)

## Pragmatic AI Labs



![alt text](https://paiml.com/images/logo_with_slogan_white_background.png)

This notebook was produced by [Pragmatic AI Labs](https://paiml.com/).  You can continue learning about these topics by:

*   Buying a copy of [Pragmatic AI: An Introduction to Cloud-Based Machine Learning](http://www.informit.com/store/pragmatic-ai-an-introduction-to-cloud-based-machine-9780134863863) from Informit.
*   Buying a copy of  [Pragmatic AI: An Introduction to Cloud-Based Machine Learning](https://www.amazon.com/Pragmatic-AI-Introduction-Cloud-Based-Learning/dp/0134863860) from Amazon
*   Reading an online copy of [Pragmatic AI:Pragmatic AI: An Introduction to Cloud-Based Machine Learning](https://www.safaribooksonline.com/library/view/pragmatic-ai-an/9780134863924/)
*  Watching video [Essential Machine Learning and AI with Python and Jupyter Notebook-Video-SafariOnline](https://www.safaribooksonline.com/videos/essential-machine-learning/9780135261118) on Safari Books Online.
* Watching video [AWS Certified Machine Learning-Speciality](https://learning.oreilly.com/videos/aws-certified-machine/9780135556597)
* Purchasing video [Essential Machine Learning and AI with Python and Jupyter Notebook- Purchase Video](http://www.informit.com/store/essential-machine-learning-and-ai-with-python-and-jupyter-9780135261095)
*   Viewing more content at [noahgift.com](https://noahgift.com/)


## Load AWS API Keys

Put keys in local or remote GDrive:  

`cp ~/.aws/credentials /Users/myname/Google\ Drive/awsml/`

### Mount GDrive




{:.input_area}
```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```


{:.output .output_stream}
```
Mounted at /content/gdrive

```



{:.input_area}
```
import os;os.listdir("/content/gdrive/My Drive/awsml")
```





{:.output .output_data_text}
```
['kaggle.json', 'credentials', 'config']
```



### Install Boto



{:.input_area}
```
!pip -q install boto3

```


### Create API Config



{:.input_area}
```
!mkdir -p ~/.aws &&\
  cp /content/gdrive/My\ Drive/awsml/credentials ~/.aws/credentials 
```


### Test Comprehend API Call



{:.input_area}
```
import boto3
comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
text = "There is smoke in San Francisco"
comprehend.detect_sentiment(Text=text, LanguageCode='en')
```





{:.output .output_data_text}
```
{'ResponseMetadata': {'HTTPHeaders': {'connection': 'keep-alive',
   'content-length': '160',
   'content-type': 'application/x-amz-json-1.1',
   'date': 'Fri, 14 Dec 2018 16:11:55 GMT',
   'x-amzn-requestid': 'fa00db36-ffba-11e8-882b-8bc33ca9084d'},
  'HTTPStatusCode': 200,
  'RequestId': 'fa00db36-ffba-11e8-882b-8bc33ca9084d',
  'RetryAttempts': 0},
 'Sentiment': 'NEUTRAL',
 'SentimentScore': {'Mixed': 0.008628507144749165,
  'Negative': 0.1037612184882164,
  'Neutral': 0.8582549691200256,
  'Positive': 0.0293553676456213}}
```



## 7.1 Sagemaker Features

### Search

#### [Demo] Search

[Manage Machine Learning Experiments with Search](https://docs.aws.amazon.com/sagemaker/latest/dg/search.html)



*   Finding training jobs
*   Rank training jobs
*   Tracing lineage of a model



### Ground Truth

![ground_truth](https://user-images.githubusercontent.com/58792/49688683-9bdba100-faca-11e8-8d93-a55ce6c35a92.png)



*   Setup and Manage labeling jobs
*   Uses active learning and human labeling
*   First 500 objects labeled per month are free



#### [Demo] Labeling Job

### Notebook

![notebooks](https://user-images.githubusercontent.com/58792/49688694-d04f5d00-faca-11e8-9fad-eb63b2534b07.png)

#### [Demo] Sagemaker Notebooks

*   Create and run Jupyter Notebooks
  -  Using Jupyter
  -  Using JupyterLab
  -  Using the terminal
  
*   Lifecycle configurations

*   Git Repositories
  - public repositories can be cloned on Notebook launch



### Training

![training](https://user-images.githubusercontent.com/58792/49688717-05f44600-facb-11e8-8d7f-cf33d272573a.png)

#### [Demo] Sagemaker Training

*   Algorithms
  -  Create algorithm
  -  Subscribe [AWS Marketplace](https://aws.amazon.com/marketplace/search/results?page=1&filters=fulfillment_options%2Cresource_type&fulfillment_options=SAGEMAKER&resource_type=ALGORITHM)

  
*   Training Jobs

*   HyperParameter Tuning Jobs


### Inference

![inference](https://user-images.githubusercontent.com/58792/49688735-2fad6d00-facb-11e8-94cb-cba9322e309b.png)

#### [Demo] Sagemaker Inference

*  Compilation jobs

*  Model packages

*  Models

*  Endpoint configurations

*  Endpoints

*  Batch transform jobs


### Built in Sagemaker Algorithms

Table of [algorithms provided by Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html)

![aws_algorithms](https://user-images.githubusercontent.com/58792/49692597-58595500-fb13-11e8-9db3-e1fe371ac36a.png)







## 7.2 DeepLense Features

![tech_specs](https://user-images.githubusercontent.com/58792/50003518-c0b39680-ff58-11e8-86dc-5a57e1482ef3.png)

![mqt](https://user-images.githubusercontent.com/58792/45307777-bfe01400-b4d4-11e8-88a3-149317f9dffc.png)

![detection](https://user-images.githubusercontent.com/58792/45308029-688e7380-b4d5-11e8-8ffb-9422184c274c.png)

#### [Demo] DeepLense

## 7.3 Kinesis Features

[Kinesis FAQ](https://aws.amazon.com/kinesis/data-streams/faqs/)

* Processes Data in Real-Time
* Can process hundreds of TBs an hour
* Example inputs are:  
 - logs
 - financial transactions
 * Streaming Data



{:.input_area}
```
!pip install -q sensible
```




{:.input_area}
```
import boto3

```




{:.input_area}
```
import asyncio
import time
import datetime
import uuid
import boto3
import json
from sensible.loginit import logger

LOG = logger(__name__)

def firehose_client(region_name="us-east-1"):
    """Kinesis Firehose client"""

    firehose_conn = boto3.client("firehose", region_name=region_name)
    extra_msg = {"region_name": region_name, "aws_service": "firehose"}
    LOG.info("firehose connection initiated", extra=extra_msg)
    return firehose_conn

async def put_record(data,
            client,
            delivery_stream_name="aws-ml-cert"):
    """
    See this:
        http://boto3.readthedocs.io/en/latest/reference/services/
        firehose.html#Firehose.Client.put_record
    """
    extra_msg = {"aws_service": "firehose"}
    LOG.info(f"Pushing record to firehose: {data}", extra=extra_msg)
    response = client.put_record(
        DeliveryStreamName=delivery_stream_name,
        Record={
            'Data': data
        }
    )
    return response


def gen_uuid_events():
    """Creates a time stamped UUID based event"""

    current_time = 'test-{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
    event_id = str(uuid.uuid4())
    event = {event_id:current_time}
    return json.dumps(event)

def send_async_firehose_events(count=100):
    """Async sends events to firehose"""

    start = time.time() 
    client = firehose_client()
    extra_msg = {"aws_service": "firehose"}
    loop = asyncio.get_event_loop()
    tasks = []
    LOG.info(f"sending aysnc events TOTAL {count}",extra=extra_msg)
    num = 0
    for _ in range(count):
        tasks.append(asyncio.ensure_future(put_record(gen_uuid_events(), client)))
        LOG.info(f"sending aysnc events: COUNT {num}/{count}")
        num +=1
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
    end = time.time()  
    LOG.info("Total time: {}".format(end - start))


```




{:.input_area}
```
send_async_firehose_events(10)
```


## 7.4 AWS Flavored Python

### Boto3

* Main Interface for working with AWS

*   Any Service in AWS can be communicated with Boto
*  If Amazon is a country this is the language



#### Communicate with S3



{:.input_area}
```
import boto3
resource = boto3.resource("s3")
resource.meta.client.download_file('testntest', 'nba_2017_endorsement_full_stats.csv',
'/tmp/nba_2017_endorsement_full_stats.csv')
```




{:.input_area}
```
!ls -l /tmp
```


{:.output .output_stream}
```
total 4
srw------- 1 root root    0 Dec 14 16:11 drivefs_ipc.0
srw------- 1 root root    0 Dec 14 16:11 drivefs_ipc.0_shell
-rw-r--r-- 1 root root 1447 Dec 14 19:06 nba_2017_endorsement_full_stats.csv

```

### Pandas

Main Datascience library for AWS and Python



*   It is assumed you know about it
*   Many study videos will show examples using it





{:.input_area}
```
import pandas as pd

df = pd.read_csv("/tmp/nba_2017_endorsement_full_stats.csv")
df.head(2)
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>TEAM</th>
      <th>SALARY_MILLIONS</th>
      <th>ENDORSEMENT_MILLIONS</th>
      <th>PCT_ATTENDANCE_STADIUM</th>
      <th>ATTENDANCE_TOTAL_BY_10K</th>
      <th>FRANCHISE_VALUE_100_MILLION</th>
      <th>ELO_100X</th>
      <th>CONF</th>
      <th>POSITION</th>
      <th>AGE</th>
      <th>MP</th>
      <th>GP</th>
      <th>MPG</th>
      <th>WINS_RPM</th>
      <th>PLAYER_TEAM_WINS</th>
      <th>WIKIPEDIA_PAGEVIEWS_10K</th>
      <th>TWITTER_FAVORITE_COUNT_1K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LeBron James</td>
      <td>Cleveland Cavaliers</td>
      <td>30.96</td>
      <td>55.0</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>12.0</td>
      <td>15.45</td>
      <td>East</td>
      <td>SF</td>
      <td>32</td>
      <td>37.8</td>
      <td>74.0</td>
      <td>37.8</td>
      <td>20.43</td>
      <td>51.0</td>
      <td>14.70</td>
      <td>5.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kevin Durant</td>
      <td>Golden State Warriors</td>
      <td>26.50</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>80.0</td>
      <td>26.0</td>
      <td>17.70</td>
      <td>West</td>
      <td>SF</td>
      <td>28</td>
      <td>33.4</td>
      <td>62.0</td>
      <td>33.4</td>
      <td>12.24</td>
      <td>51.0</td>
      <td>6.29</td>
      <td>1.43</td>
    </tr>
  </tbody>
</table>
</div>
</div>



#### Descriptive Statistics with Pandas



{:.input_area}
```
df.describe()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SALARY_MILLIONS</th>
      <th>ENDORSEMENT_MILLIONS</th>
      <th>PCT_ATTENDANCE_STADIUM</th>
      <th>ATTENDANCE_TOTAL_BY_10K</th>
      <th>FRANCHISE_VALUE_100_MILLION</th>
      <th>ELO_100X</th>
      <th>AGE</th>
      <th>MP</th>
      <th>GP</th>
      <th>MPG</th>
      <th>WINS_RPM</th>
      <th>PLAYER_TEAM_WINS</th>
      <th>WIKIPEDIA_PAGEVIEWS_10K</th>
      <th>TWITTER_FAVORITE_COUNT_1K</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.216000</td>
      <td>21.700000</td>
      <td>99.800000</td>
      <td>80.200000</td>
      <td>21.375000</td>
      <td>15.678000</td>
      <td>29.300000</td>
      <td>33.890000</td>
      <td>70.800000</td>
      <td>33.890000</td>
      <td>11.506000</td>
      <td>44.100000</td>
      <td>6.532000</td>
      <td>2.764000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.294438</td>
      <td>15.362653</td>
      <td>2.394438</td>
      <td>5.202563</td>
      <td>8.507554</td>
      <td>1.361142</td>
      <td>3.164034</td>
      <td>2.303837</td>
      <td>8.390471</td>
      <td>2.303837</td>
      <td>6.868487</td>
      <td>12.591443</td>
      <td>5.204233</td>
      <td>3.646399</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.110000</td>
      <td>8.000000</td>
      <td>94.000000</td>
      <td>70.000000</td>
      <td>10.250000</td>
      <td>13.740000</td>
      <td>24.000000</td>
      <td>29.900000</td>
      <td>60.000000</td>
      <td>29.900000</td>
      <td>1.170000</td>
      <td>26.000000</td>
      <td>2.690000</td>
      <td>0.350000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.707500</td>
      <td>13.000000</td>
      <td>100.000000</td>
      <td>78.500000</td>
      <td>13.125000</td>
      <td>15.250000</td>
      <td>28.000000</td>
      <td>32.725000</td>
      <td>62.500000</td>
      <td>32.725000</td>
      <td>6.015000</td>
      <td>32.500000</td>
      <td>3.402500</td>
      <td>0.865000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>23.880000</td>
      <td>14.500000</td>
      <td>100.000000</td>
      <td>80.500000</td>
      <td>22.500000</td>
      <td>15.450000</td>
      <td>28.000000</td>
      <td>33.850000</td>
      <td>73.000000</td>
      <td>33.850000</td>
      <td>12.860000</td>
      <td>46.500000</td>
      <td>4.475000</td>
      <td>1.485000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.500000</td>
      <td>31.250000</td>
      <td>100.000000</td>
      <td>83.250000</td>
      <td>26.000000</td>
      <td>16.275000</td>
      <td>31.750000</td>
      <td>34.975000</td>
      <td>77.750000</td>
      <td>34.975000</td>
      <td>16.890000</td>
      <td>51.000000</td>
      <td>5.917500</td>
      <td>2.062500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.960000</td>
      <td>55.000000</td>
      <td>104.000000</td>
      <td>89.000000</td>
      <td>33.000000</td>
      <td>17.700000</td>
      <td>35.000000</td>
      <td>37.800000</td>
      <td>81.000000</td>
      <td>37.800000</td>
      <td>20.430000</td>
      <td>65.000000</td>
      <td>17.570000</td>
      <td>12.280000</td>
    </tr>
  </tbody>
</table>
</div>
</div>



### Plotting with Python



{:.input_area}
```
import warnings
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)
```




{:.input_area}
```
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(11, 8.5)
sns.regplot(data=df, 
            x="SALARY_MILLIONS", y="ENDORSEMENT_MILLIONS", 
            order=2).set_title("NBA Salary & Endorsements")
```





{:.output .output_data_text}
```
Text(0.5,1,'NBA Salary & Endorsements')
```




{:.output .output_png}
![png](../images/lessons/Lesson7_AWSML_Case_Studies_58_1.png)



### Putting it all together (Production Style)



{:.input_area}
```
!pip -q install python-json-logger
```




{:.input_area}
```
import logging
from pythonjsonlogger import jsonlogger

LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
LOG.addHandler(logHandler)

import click
import boto3
import pandas as pd

TEST_DF = pd.DataFrame(
    {"SentimentRaw": ["I am very Angry",
                    "We are very Happy",
                    "It is raining in Seattle"]}
)

def create_sentiment(row):
    """Uses AWS Comprehend to Create Sentiments on a DataFrame"""

    LOG.info(f"Processing {row}")
    comprehend = boto3.client(service_name='comprehend', region_name="us-east-1")
    payload = comprehend.detect_sentiment(Text=row, LanguageCode='en')
    LOG.debug(f"Found Sentiment: {payload}")    
    sentiment = payload['Sentiment']
    return sentiment

def apply_sentiment(df, column="SentimentRaw"):
    """Uses Pandas Apply to Create Sentiment Analysis"""

    df['Sentiment'] = df[column].apply(create_sentiment)
    return df
```




{:.input_area}
```
df = apply_sentiment(TEST_DF)
```




{:.input_area}
```
df.head()
```





<div markdown="0" class="output output_html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SentimentRaw</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I am very Angry</td>
      <td>NEGATIVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>We are very Happy</td>
      <td>POSITIVE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>It is raining in Seattle</td>
      <td>NEUTRAL</td>
    </tr>
  </tbody>
</table>
</div>
</div>



## 7.5 Cloud9



*   Web based [development environment](https://docs.aws.amazon.com/cloud9/latest/user-guide/sample-python.html)


![Cloud9](https://user-images.githubusercontent.com/58792/50033517-afe14000-ffad-11e8-894a-f70389a046d7.png)



### [Demo]Cloud9



*   Create a cloud9 environment
*   Install Python 3.6

---



```bash
sudo yum -y update
sudo yum -y install python36

```







**Python Lambda Function**

```python
import json
import decimal


def lambda_handler(event, context):

  print(event)
  if 'body' in event:
    event = json.loads(event["body"])
  
  amount = float(event["amount"])
  res = []
  coins = [1,5,10,25]
  coin_lookup = {25: "quarters", 10: "dimes", 5: "nickels", 1: "pennies"}
  coin = coins.pop()
  num, rem  = divmod(int(amount*100), coin)
  res.append({num:coin_lookup[coin]})
  while rem > 0:
    coin = coins.pop()
    num, rem = divmod(rem, coin)
    if num:
      if coin in coin_lookup:
        res.append({num:coin_lookup[coin]})

  response = {
    "statusCode": "200",
    "headers": { "Content-type": "application/json" },
    "body": json.dumps({"res": res})
  }

  return response
```



**payload**



```javascript
{"amount": ".71"}
```



**response**



```javascript
Response
{
    "statusCode": "200",
    "headers": {
        "Content-type": "application/json"
    },
    "body": "{\"res\": [{\"2\": \"quarters\"}, {\"2\": \"dimes\"}, {\"1\": \"pennies\"}]}"
}

Function Logs
{'amount': '.71'}

Request ID
d7ec2cad-8da0-4394-957e-41f07bad23ae
```




## 7.6 Key Terminology

### Sagemaker Built-in Algorithms



#### BlazingText


* unsupervised learning algorithm for generating **Word2Vec embeddings.**
* aws blog post [BlazingText](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-blazingtext-parallelizing-word2vec-on-multiple-cpus-or-gpus/)



![BlazingText](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2018/01/18/sagemaker-word2vec-3-1.gif)

#### DeepAR Forecasting

* supervised learning algorithm for forecasting scalar (that is, one-dimensional) time series using recurrent neural networks (RNN)
* [DeepAR Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)

![DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/images/deepar-2.png)
