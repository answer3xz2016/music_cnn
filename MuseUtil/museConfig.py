import os

logDebug = False         # whether to print some debug info

myMuseBase = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
myKeyDir = myMuseBase + '/WebUI/keys'
myTrainedModel = myMuseBase + '/TrainedModels'
myPostgresKey = myKeyDir + "/.credentials.cred"
myPostgresKeyXimalaya = myKeyDir + "/.credentials_ximalaya.cred"
myPostgresKeyLastFm = myKeyDir + "/.credentials_lastfm.cred"
sevendigitalKeyFile = myKeyDir + "/7digital.txt"
sevendigitalKey = ''
myBestCFModel_Umf = myTrainedModel + "/CCF-muse-1M-step8_rank_10_10_lambda_1.0_1.0_alpha_0.0001_0.0001_iterations_50_50.log_Umf.npy"
myBestCFModel_Inf = myTrainedModel + "/CCF-muse-1M-step8_rank_10_10_lambda_1.0_1.0_alpha_0.0001_0.0001_iterations_50_50.log_Inf.npy"

myXimalayaKeyDir = myMuseBase + '/MuseMusicVendorAPI/keys'
myXimalayaKeyFile = myXimalayaKeyDir + '/ximalaya_key.txt'
myXimalayaSignatureGeneratorDir = myMuseBase + '/MuseMusicVendorAPI/Signature_Generator'
myMacAddress = '98:01:a7:95:ff:dd'
