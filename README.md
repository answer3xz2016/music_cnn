[![build status](https://githost.nevint.com/data-science/muse/badges/master/build.svg)](https://githost.nevint.com/data-science/muse/commits/master) [![coverage report](https://githost.nevint.com/data-science/muse/badges/master/coverage.svg)](https://githost.nevint.com/data-science/muse/commits/master)
========================================================
========================================================

Muse 缪斯 project
Author: Dr. Zhou Xing 
Team: Data Science

========================================================
========================================================


This is the production code for the Muse project, which
includes various models such as C.C.F, training of the
model, making inference or predictions. Web UI service
REST API to provide recommendations.

========================================================
========================================================


========================================================
========================================================

Bench mark results and performances on MSD

1. Training Collaborative Filtering on Million Song 
   Database (MSD), number of users: 109970 and number of 
   songs: 145868, total number of ratings ~ 1.5 M

2. Bench mark performances: 

RMSE on absolute play counts (0 - ~ 900) =  2.44
RMSE on absolute play counts = 2.44 as compared to the
average rating trivial prediction RMSE = 6.94

AUC on binarized play counts = 0.89 after selecting only
users who has listened to more than 50 songs


RMSE on binary scale implicit rating (use play counts as 
confidence level) =  0.795

AUC on binary scale implicit rating (use play counts as
confidence level) = 0.67




Training Time = 39 minutes for 50 epochs @ SPARK cluster 
                of 4 worker node, HDFS cluster of 4 
                datanodes, explicit rating ( 1-5 rating)

	      = 1.8 hour for 50 epochs @ SPARK cluster
                of 4 worker node, HDFS cluster of 4
                datanodes, implicit rating (play counts)


========================================================
========================================================


Bench mark results on performances on Yahoo music 
 (explicit rating):

There are int total ~700 M ratings coming from 200,000 
users to 136,736 songs, the data is partitioned into 10 
chunks, here I only use 1 chunk for case study: that's 
~ 70 Million ratings from 200,000 users to 136,736 songs

Testing on 2 M ratings, number of users: 200,000 and 
number of songs: 127,771, RMSE = 1.32


RMSE on rating from 1 - 5 =  1.32

AUC for 50k user sub-sample = 0.79
AUC for 40k user sub-sample = 0.78
AUC for 30k user sub-sample = 0.74
AUC for 20k user sub-sample = 0.66
AUC for 10k user sub-sample = 0.64


========================================================
========================================================





========================================================
========================================================

Some instructions of example use cases:

1. Training using SPARK cluster 
   1.1 Train C.C.F model:
    - $SPARK_HOME/bin/spark-submit --driver-class-path /home/ec2-user/muse/JDBCDriver/postgresql-9.4.1208.jre7.jar Examples/train_CCF.py <step> &   
    - note: make sure the JDBC driver is installed at each spark worker node

   1.2 Train SPARK C.F. model: 
    - $SPARK_HOME/bin/spark-submit --driver-class-path /home/ec2-user/muse/JDBCDriver/postgresql-9.4.1208.jre7.jar Models/SPARK_CF.py &

   1.3 Train C.F model with data in HDFS (yahoo music), now we do not need JDBC anymore to read data from SQL datastore
    -

2. Computing RMSE

3. Plotting cost function learning curve

4. Browsing MillionSong database:
   - git clone git@nextev.githost.io:data-science/muse.git
   - cd muse
   - switch to dev branch : git checkout dev
   - point myMuseBase to your file system (Muse home directory): edit MuseUtil/museConfig.py myMuseBase='/home/ec2-user/muse' for example 
   - install Muse: python setup.py install
   - run example script to browse data in RDS: python Examples/browse_million_song.py

5. Plot the clutering of Ximalaya podcast items using content-based model where content is extracted from text features.
   - python MuseModels/ContentBased.py , this script will load word2vec embeddings and perform dimensionality reduction and plot the distribution of musics in the latent space