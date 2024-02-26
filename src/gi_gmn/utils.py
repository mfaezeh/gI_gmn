import numpy as np
import pandas as pd
import nibabel as nib

def get_subject(path1 ,path2, masker, TR, NRecord,ROI):  # two record version
    file_GM = nib.load(path1)             
    file_data1 = nib.load(path1).get_fdata()[:, :, :, :TR]
    file_data2 = nib.load(path2).get_fdata()[:, :, :, :TR]
    
    ins1, ins2, ins3 = file_data1.shape[0], file_data1.shape[1], file_data1.shape[2]
    # chekc larg time points
    # if file_data1.shape[3] >170 or file_data2.shape[3] > 170:
    #     print('one')
        
    ins = ins1 * ins2 * ins3
    print(file_data1.shape)

    file_data1 = file_data1.reshape(ins, TR)
    file_data2 = file_data2.reshape(ins, TR)
    all_ = np.concatenate((file_data1, file_data2), axis = 1)

    file_data = all_.reshape((ins1, ins2, ins3, TR * NRecord))
    print(file_data.shape, all_.shape)

    fileo = nib.Nifti1Image(file_data, file_GM.affine)  
    DMN_signals = masker.fit_transform(fileo)
    # print(DMN_signals.shape,NR)
    
    #Normalize time series of each ROI independetly with mean of one and std of zero
    for i in range(ROI):
        mu = np.mean(DMN_signals[:, i], axis = 0)
        std = np.std(DMN_signals[:, i], axis = 0)
        DMN_signals[:, i] = (DMN_signals[:, i] - mu) / std
    
    return DMN_signals

def  partial_corr(pre, ROI):
    norm = np.zeros((ROI, ROI))
    
    # Normalize to get Partial correlation -1.0 < < 1.0
    for i in range(ROI):
        for j in range(ROI):
            norm[i,j] = (-1 * pre[i,j]) / np.sqrt(pre[i,i] * pre[j,j])
    return norm

def flatten_matrix(d_):
     d = pd.DataFrame(d_)
     d_masked = d.mask(np.tril(np.ones(d.shape)).astype(np.bool))
     #                 print(d_masked)
     d_masked = d_masked[abs(d_masked) >= 0].stack().reset_index()
     #                 print(d_masked.shape)
     #                 print(type(d_masked))

     # to only consider the last column for the feature set * feature meaning here *
     #                 print(d_masked.columns)
     d_masked = d_masked [[0]]
     #             print(d_masked[0].values.tolist())
     corr_list = d_masked[0].values.tolist()
     corr_np = np.array(corr_list)
     # print(corr_np.shape)
     return corr_np

def get_seed_id(str_, all):
    all = np.array(all)
#     print(all)
    id_str = np.where(all == str_)[0][0]
#     print(id_str)
    d_ = np.ones((all.shape[0], all.shape[0]))
    d = pd.DataFrame(d_)
    d_masked = d.mask(np.tril(np.ones(d.shape)).astype(np.bool))
    # print(d_masked)
    d_masked = d_masked[abs(d_masked) >= 0].stack().reset_index()
    level0 = d_masked['level_0'].values
    level1 = d_masked['level_1'].values

    d_masked = d_masked [[0]]

    corr_list = d_masked[0].values.tolist()

    list_str0 = np.where( level0 == id_str)[0]
    list_str1 = np.where(level1 == id_str)[0]

    return np.concatenate((list_str0, list_str1))

def get_labels(i, arr):
    all_data = np.array(arr)
    d_ = np.ones((len(arr), len(arr)))
    d = pd.DataFrame(d_)
    d_masked = d.mask(np.tril(np.ones(d.shape)).astype(np.bool))
    # print(d_masked)
    d_masked = d_masked[abs(d_masked) >= 0].stack().reset_index()
    level0 = d_masked['level_0'].values
    level1 = d_masked['level_1'].values

    x = level0[i]
    y = level1[i]
    # print(x, type(all[x] ))
    corr_pair = all_data[x] + " " + all_data[y]

    d_masked = d_masked [[0]]
    #             print(d_masked[0].values.tolist())
    corr_list = d_masked[0].values.tolist()

    return corr_pair

# TODO: The return statement only activates after the if, is that correct?

def get_correlation_indexes(str1, str2, all_data):
    fg = get_seed_id(str1, all)#,get_seed_id('Precuneus_R')
    for i in range(len(fg)):
        # print(get_lables(fg[i]))
        if get_labels(fg[i], all_data) == str1 + " " + str2 or get_labels(fg[i], all_data) == str2 + " " + str1:
            # print(fg[i])
            return fg[i]

def split_into_chunks(arr, chunk_size):
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

def degree_sparsity(arr):
    size = (arr.shape[0] - 1) * (arr.shape[0] / 2)
    return np.count_nonzero(arr) / (2 * size)

def roi_conn(copy_norm):
    roi,conn = [],[]
    for i in range(np.nonzero(copy_norm)[0].shape[0]):
        conn1 = str(np.nonzero(copy_norm)[0][i])+ "_" + str(np.nonzero(copy_norm)[1][i])
        conn2 = str(np.nonzero(copy_norm)[1][i])+ "_" + str(np.nonzero(copy_norm)[0][i])
        if np.nonzero(copy_norm)[0][i] not in roi:
            roi.append(np.nonzero(copy_norm)[0][i])
        if np.nonzero(copy_norm)[1][i] not in roi:
            roi.append(np.nonzero(copy_norm)[1][i])
        if (conn1  not in conn )and (conn2 not in conn) and (conn2!=conn1): # removing two way and self connection ( totall - 120)/2
            conn.append(conn1)
    return roi,conn

def get_inx(str1,str2,all):
    fg = get_seed_id(str1,all)#,get_seed_id('Precuneus_R')
    for i in range(len(fg)):
        # print(get_lables(fg[i]))
        if get_labels(fg[i],all) == str1+" "+str2 or get_labels(fg[i],all) == str2+" "+str1:
            # print(fg[i])
            return fg[i]
        
def get_recall(n,l,cnt):
    c=0
    totall = np.count_nonzero(l == cnt)
    for i in range(n.shape[0]):
        if l[i] == cnt and n[i] == cnt:
            c = c + 1
    # print(c,totall)
    return c/totall
        
