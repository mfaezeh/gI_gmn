import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.covariance import GraphicalLasso
import gi_gmn.utils as giu

class GroupSubjectGMN:
    def __init__(
            self, 
            NC, 
            NP, 
            NC_test, 
            TR,
            NP_test, 
            NRecord, ROI
            ):
           
        self.NC = NC
        self.NP = NP
        self.TR = TR
        self.NC_test = NC_test
        self.NP_test = NP_test
        self.ROI = ROI
        self.NRecord = NRecord

        return
    
    def group_level(self, patient_data, control_data, alpha_p, alpha_c, diff, TR_P):

        if not diff:
            TR_P = self.TR

        print("Model: Group-level GMN")

        a_p , a_c = [],[]
        runs = 5

        for i in range(runs):

            #randomly select controls for test
            index_control = [np.random.choice(np.arange(0 , self.NC - 1), size = self.NC_test,replace = False)]
            control = [j for j in range(self.NC)]
            index_control_test = list(index_control[0])
            index_control_train = list(set(control)  - set(index_control_test))

            #randomly select patients for test
            index_patient = [np.random.choice(np.arange(0 , self.NP - 1), size = self.NP_test,replace = False)]
            patient = [j for j in range(self.NP)]
            index_patient_test = list(index_patient[0])
            index_patient_train = list(set(patient)  - set(index_patient_test))


            parts_c = giu.split_into_chunks(control_data, self.TR * self.NRecord)
            parts_p = giu.split_into_chunks(patient_data, TR_P * self.NRecord)


            control_test_data = np.array([part for i,part in enumerate(parts_c) if i in index_control_test ])\
            .reshape(self.NC_test * self.TR * self.NRecord, self.ROI)
            control_train_data = np.array([part for i,part in enumerate(parts_c) if i in index_control_train ])\
            .reshape( (self.NC - self.NC_test) * self.TR * self.NRecord, self.ROI)

            patient_test_data = np.array([part for i,part in enumerate(parts_p) if i in index_patient_test ])\
            .reshape(self.NP_test * TR_P * self.NRecord, self.ROI)
            patient_train_data = np.array([part for i,part in enumerate(parts_p) if i in index_patient_train ])\
            .reshape((self.NP - self.NP_test) * TR_P * self.NRecord, self.ROI)

            

            # get the distribution and Convatraince matrix of the model
            patientGL = GraphicalLasso(alpha = alpha_p,max_iter = 350)
            controlGL = GraphicalLasso(alpha = alpha_c,max_iter = 360)


            # train the model
            patient_model = patientGL.fit(patient_train_data)
            control_model = controlGL.fit(control_train_data)


            cov_patient = np.around(patient_model.precision_, decimals=3)
            cov_control = np.around(control_model.precision_, decimals=3)

            # print(degree_sparsity(cov_control),degree_sparsity(cov_patient))

            #save the precision matrix - group level
            # cov_control = pd.DataFrame(cov_control)
            # cov_control.to_csv('../../Ataxia/control_precision.csv',index = False)
            # cov_patient = pd.DataFrame(cov_patient)
            # cov_patient.to_csv('../../Ataxia/patient_precision.csv',index = False)

            #test
            recall_p = 0 
            recall_c = 0

            for indx in range(0, patient_test_data.shape[0], TR_P * self.NRecord):
                log_prob_control = controlGL.score(patient_test_data[indx:indx + TR_P * self.NRecord,:])
                log_prob_patient = patientGL.score(patient_test_data[indx:indx + TR_P * self.NRecord,:])
                # print(log_prob_patient-log_prob_control)
                if (log_prob_control < log_prob_patient):
                    recall_p = recall_p + 1
                    
            for indx in range(0, control_test_data.shape[0], self.TR * self.NRecord):
                log_prob_control = controlGL.score(control_test_data[indx:indx + self.TR * self.NRecord, :])
                log_prob_patient = patientGL.score(control_test_data[indx:indx + self.TR * self.NRecord, :])
                # print(log_prob_patient-log_prob_control)
                if (log_prob_control > log_prob_patient):
                    recall_c = recall_c + 1


            recall_p = recall_p / self.NP_test
            recall_c = recall_c / self.NC_test
            a_p.append(recall_p)
            a_c.append(recall_c)

        print(np.array(a_p).mean(),np.array(a_p).std(), np.array(a_c).mean(), np.array(a_c).std())

        return cov_patient, cov_control

    def prune_connections(self, pre_p, pre_c, thr_weak, Thresholding_diff):
        # Normalize the precision matrix for visualizing the SCA-HCP goup level models

        #STEP 1/3 PREPROCESSING: normalize the precision matrix with z tranformation 

        norm_p = np.zeros((self.ROI,self.ROI))
        norm_c = np.zeros((self.ROI,self.ROI))

        # Normalize to get Partial correlation -1.0 < < 1.0
        for i in range(self.ROI):
            for j in range(self.ROI):
                norm_c[i,j] = (-1 * pre_c[i,j]) / np.sqrt(pre_c[i,i]*pre_c[j,j])
                norm_p[i,j] = (-1 * pre_p[i,j]) / np.sqrt(pre_p[i,i]*pre_p[j,j])

        # #save the normalized precision matrix
        # cov_control = pd.DataFrame(norm_c)
        # cov_control.to_csv('./Ataxia/normpre_c.csv',index = False)
        # cov_patient = pd.DataFrame(norm_p)
        # cov_patient.to_csv('./Ataxia/normpre_p.csv',index = False)  

        # STEP 2/3 - PREPROCESSING: precision matrix analysis - remomving weak connections
        copy_norm_p = np.zeros((self.ROI,self.ROI))
        copy_norm_c = np.zeros((self.ROI,self.ROI))

        for i in range(self.ROI):
            for j in range(self.ROI):
                if norm_p[i,j] > thr_weak or norm_p[i,j]< (-1*thr_weak):
                    copy_norm_p[i,j] = norm_p[i,j]


        for i in range(self.ROI):
            for j in range(self.ROI):            
                if norm_c[i,j] > thr_weak or norm_c[i,j]< (-1*thr_weak):
                    copy_norm_c[i,j] = norm_c[i,j]

        # # STEP 3/3 PREPROCESSING: apply the thresholding on your the model (for representation)

        for i in range(self.ROI):
            for j in range(self.ROI):
                diff = np.abs(np.abs(copy_norm_p[i,j]) - np.abs(copy_norm_c[i,j]))
                # if copy_norm_c[i,j]!=0 and copy_norm_c[i,j]!=0:
                if diff < Thresholding_diff:
                    copy_norm_p[i,j] = 0.0
                    copy_norm_c[i,j] = 0.0

        return copy_norm_p, copy_norm_c
    
    
    def extract_connections_regions(self, copy_norm_p, copy_norm_c, all_col):
        # identifying unique regions and connections

        roi_c,conn_c = giu.roi_conn(copy_norm_c)
        roi_p,conn_p = giu.roi_conn(copy_norm_p)

        # Union 
        modelp = np.nonzero(copy_norm_p)
        modelc = np.nonzero(copy_norm_c)
        regions_p = np.union1d(modelp[0],modelp[1])
        regions_c = np.union1d(modelc[0],modelc[1])

        regions = np.union1d(regions_c,regions_p)

        connU = np.union1d(np.array(conn_p),np.array(conn_c))
        roiU =  np.union1d(np.array(roi_p),np.array(roi_c))
        # print(len(conn_p),len(roi_p),len(conn_c),len(roi_c),connU.shape,connU,regions.shape )#roiU.shape,roiU
        # regions36 = np.array(all_col)[regions]

        i,j,val_p,val_c = [],[],[],[]
        for conn in connU:
            i.append(all_col[int(conn.split("_")[0])])
            j.append(all_col[int(conn.split("_")[1])])
            valp = copy_norm_p[int(conn.split("_")[1]),int(conn.split("_")[0])]
            valc = copy_norm_c[int(conn.split("_")[1]),int(conn.split("_")[0])]
            val_p.append(valp)
            val_c.append(valc)
            # print( all_col[i],all_col[j], copy_norm_p[j,i],copy_norm_c[i,j])

        df_CONN = pd.DataFrame()
        df_CONN["ROI-1"] = i
        df_CONN["ROI-2"] = j
        df_CONN["SCA"] = val_p
        df_CONN["HCP"] = val_c
        df_CONN["diff"] = np.abs(np.array(val_c) - np.array(val_p))
        df_CONN = df_CONN.sort_values(by = "diff",ascending = False)
        df_CONN = df_CONN.drop(columns  = ['diff'])
        df_CONN.index = np.arange(connU.shape[0]) + 1
        # print(df_CONN.round(3).to_latex())
        # df_CONN
        return df_CONN, regions, connU
    
    def individual_level(self, patient_data, control_data, alpha_p, alpha_c, regions, diff, TR_P):
        #Individual level precision matrix on selected connectivity of HCP-SCA group level model %

        if not diff:
            TR_P = self.TR

        c,p = 0, 0
        
        CORRL = int((regions.shape[0] * (regions.shape[0]-1)) / 2 )

        all_pre_c, all_pre_p= np.zeros((self.NC, CORRL)), np.zeros((self.NP,CORRL))

        for i in range(0,self.TR * self.NRecord * self.NC, self.TR * self.NRecord):

            controlGL = GraphicalLasso(alpha = alpha_c,max_iter = 360)
            # train the model
            
            control_model = controlGL.fit(control_data[i:i + self.TR * self.NRecord,regions])
            cov_control = np.around(control_model.precision_, decimals=3)
            print(cov_control.shape)

            cov_control_ = giu.partial_corr(cov_control, regions.shape[0])
            # cov_control_ = np.zeros((self.ROI, self.ROI))
    
            # # Normalize to get Partial correlation -1.0 < < 1.0
            # for i in range(self.ROI):
            #     for j in range(self.ROI):
            #         cov_control_[i,j] = (-1 * cov_control[i,j]) / np.sqrt(cov_control[i,i] * cov_control[j,j])

            all_pre_c[c,:] = giu.flatten_matrix(cov_control_)
            c = c + 1

        for k in range(0, TR_P * self.NP * self.NRecord,TR_P * self.NRecord):
            patientGL = GraphicalLasso(alpha = alpha_p,max_iter = 360)

            # train the model
            patient_model = patientGL.fit(patient_data[k:k + TR_P * self.NRecord,regions])
            cov_patient = np.around(patient_model.precision_, decimals=3)


            cov_patient_ = giu.partial_corr(cov_patient, regions.shape[0])
            # cov_patient_ = np.zeros((self.ROI, self.ROI))
    
            # # Normalize to get Partial correlation -1.0 < < 1.0
            # for i in range(self.ROI):
            #     for j in range(self.ROI):
            #         cov_patient_[i,j] = (-1 * cov_patient[i,j]) / np.sqrt(cov_patient[i,i] * cov_patient[j,j])

            all_pre_p[p,:] = giu.flatten_matrix(cov_patient_)
            p = p + 1

        for ij in range(CORRL):
            all_pre_c[:,ij][all_pre_c[:,ij] == -0.0] = 0.0
            all_pre_p[:,ij][all_pre_p[:,ij] == -0.0] = 0.0

        #save the precision matrix
        # cov_control = pd.DataFrame(all_pre_c)
        # cov_control.to_csv('./Ataxia/Individual_precision_control.csv',index = False) #496*36 
        # cov_patient = pd.DataFrame(all_pre_p)
        # cov_patient.to_csv('./Ataxia/Individual_precision_patient.csv',index = False) #496 * 42 

        return all_pre_p, all_pre_c
    
    def mask_CONN(self, all_pre_p, all_pre_c, all_col, regions, connU):
        #Feature selection based on Union of connectivity of SCA-HCP group level model     

        inds = [] # ids of connections in conn
        all_col13 = np.array(all_col)[regions]

        for conn in connU:
            gg = conn.split("_")[0]
            # print(all_col[int(gg)],conn)#np.array(all_col)[gg])
            indx = giu.get_inx(all_col[int(conn.split("_")[0])] ,all_col[int(conn.split("_")[1])],all_col13)
            inds.append(indx)

        cov_control = pd.DataFrame(all_pre_c[:,inds])
        # cov_control.to_csv('./Modafiline/all_precision_pre_0.010.03.csv',index = False) #496*36 
        cov_patient = pd.DataFrame(all_pre_p[:,inds])
        # cov_patient.to_csv('./Modafiline/all_precision_post_0.010.03.csv',index = False) #496 * 42 

        return cov_patient, cov_control


    def binary_classfier(self, masked_conn_patient, masked_conn_control, application, c_lr):

        run = 20
        models =  ['svmrbf','lr','svmlinear','knn','rf']

        # the code be used for train or test
        # application = "test" #train

        if application == "test":
            np_ = self.NP - self.NP_test
            nc_ = self.NC - self.NC_test
            ap,ac = self.NP, self.NC
        else:
            np_ = self.NP_test
            nc_ = self.NC_test
            ap,ac = self.NP - self.NP_test, self.NC - self.NC_test


        # for trian or validation amaonge 30 SCA we use 21 for train 9 for test and for HCP amone 25 we use 17 for train and 8 for test
    

        for model in models:
            print(model)
            print(np_, nc_,ap,ac)

            rp_test, rc_test, rp_train, rc_train,pre,acc,coef = [], [], [], [],[],[],[]
            cX_keep, cXtest_keep = [], []
            

            for i in range(run):

                # np.random.shuffle(all_pre_p) 
                # np.random.shuffle(all_pre_c) 

                # permTrain = np.random.permutation(55) # if train 20 + 17
                # X = np.concatenate((all_pre_p[:30,:], all_pre_c[:25,:]))  #55
                # Y = np.concatenate(( np.ones(30) , np.zeros(25))) # 12 + 11  # train hyperparemter from 30 - 25 to 20 17
                # X = X[permTrain]
                # Y = Y[permTrain]

                # X = X[:,inds]

                # permTest = np.random.permutation(11+12)#(NP_test + NC_test)
                # X_test = np.concatenate((all_pre_p[30:,:], all_pre_c[25:,:]))#((all_pre_p[20:30,:], all_pre_c[17:25,:])) 
                # Y_test = np.concatenate(( np.ones(12) , np.zeros(11))) 
                # X_test = X_test[permTest]
                # Y_test = Y_test[permTest]

                # X_test = X_test[:,inds]

                permTestP = random.sample(range(0,ap),ap-np_)
                permTestC = random.sample(range(0,ac),ac-nc_)

                permTrainP = [i for i in np.arange(ap) if i not in permTestP] 
                permTrainC =  [i for i in np.arange(ac) if i not in permTestC]


                X = np.concatenate((np.array(masked_conn_patient)[permTrainP],np.array(masked_conn_control)[permTrainC]))
                Y = np.concatenate(( np.ones(np_) , np.zeros(nc_))) # 12 + 11
                # X = X[:,inds]
                X_test = np.concatenate((np.array(masked_conn_patient)[permTestP],np.array(masked_conn_control)[permTestC])) 
                Y_test = np.concatenate(( np.ones(ap-np_) , np.zeros(ac-nc_)))
                # X_test = X_test[:,inds] 
                cX_keep.append(X)
                cXtest_keep.append(X_test)

                if model == 'svmrbf':
                    clf = SVC(kernel='rbf',C=0.06,gamma=0.009)
                elif model == 'svmlinear' :
                    clf = SVC(kernel='linear')#)
                elif model == 'knn' :
                    clf =  KNeighborsClassifier(5)
                elif model == 'rf':
                    clf = RandomForestClassifier(max_depth=6, random_state=0)
                else:
                    clf = LogisticRegression(random_state=0,penalty='l2',solver='liblinear',C=c_lr)
                clf.fit(X, Y)
                
            #     pre.append(accuracy_score(Y_test, clf.predict(X_test)))
            
            # print(np.mean(pre))

                rp_test.append(giu.get_recall(clf.predict(X_test),Y_test,1))
                rc_test.append(giu.get_recall(clf.predict(X_test),Y_test,0))

                rp_train.append(giu.get_recall(clf.predict(X),Y,1))
                rc_train.append(giu.get_recall(clf.predict(X),Y,0))

                acc.append((accuracy_score(Y_test,clf.predict(X_test))))
                # coef.append(clf.coef_)
            coef = np.array(coef)
            print(np.array(rp_test).mean(),np.array(rc_test).mean(), np.array(rp_train).mean(),np.array(rc_train).mean() ,np.array(acc).mean())
            print(np.array(rp_test).std(),np.array(rc_test).std(), np.array(rp_train).std(),np.array(rc_train).std() ,np.array(acc).std())


        return 

    def fit(self):
        self.group_level()
        self.prune_connections()
        self.individual_level()
        print("Model fitted successfully!")

