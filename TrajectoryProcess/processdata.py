import os
import time

import pandas as pd
import numpy as np
from numpy import linalg as LA
import _pickle as pickle

global baseFolder



def setBaseFolder(folder=''):
    global baseFolder
    
    if folder=='':
        baseFolder = os.getcwd()
    else:
        baseFolder = folder
        
    if baseFolder[-1] != '/':
        baseFolder = baseFolder+'/'

def getBaseFolder():
    global baseFolder
    return baseFolder


def getAllDataFilteredFileName(code,min_trajectory_duration):
    fileName = code + '-over-' + str(min_trajectory_duration) + '.txt'
    fileDir = getDataFolder()
    return fileDir + fileName


def getResultsFolder(code):
    '''
    Returns absolute path for results folder
    '''
    return getCodeFolder('results/'+code)

def getTempFolder(code):
    '''
    Returns absolute path for temporal pickle folder
    '''
    return getCodeFolder('temp/'+code)

def getDataFolder():
    '''
    Returns absolute path for data storage folder
    '''
    return getCodeFolder('data')

def getCodeFolder(prefix):
    '''
    Each raw dataset should have different folders, as traj_id counters are local
    '''
    base_folder = getBaseFolder()
    if base_folder[-1] != '/':
        base_folder = base_folder+'/'
    final_folder = base_folder+prefix+'/'
    try:
        os.mkdir(final_folder)
    except OSError:
        pass
    return final_folder

def getTrajectoriesFileName(code):
    '''
    File where the trajectory ids are stored. All matrix and vector indexes refer to this one.
    '''
    return getResultsFolder(code)+'trajectoriesIDs.csv'

def getTempDistFileName(id_i, n_digits,code):
    '''
    When calculating minimum distances between trajectories, 
    this returns the file where distances between traj number id_i and n_tr are.
    '''
    padded_number = str(id_i).zfill(n_digits)
    fileName = getTempFolder(code)+'dist_dict-' + padded_number + '.pickle'
    return fileName

def getTrajsFileName(code,id_i, n_digits):
    '''
    Returns the temporal file where the dataframe for traj id_i is stored.
    '''  
    pickleFolder = getTempFolder(code)
    padded_number = str(id_i).zfill(n_digits)
    fileName = pickleFolder+'traj_df-' + padded_number + '.pickle'
    return fileName


def saveDistDict(id_i, n_digits,code, last_entry, ddict):
    temp_results_filename = getTempDistFileName(id_i, n_digits,code)
    data = (last_entry, ddict)
    pickle.dump(data, open(temp_results_filename, "wb"))

def openIDFile(code,id_i, n_digits):
    traj_fileName = getTrajsFileName(code,id_i, n_digits)
    id_i_data = []
    try:
        with open(traj_fileName, 'rb') as fp:
            id_i_data = pickle.load(fp)
    except IOError:
        print("Couldn't open file (%s)" % (traj_fileName))
    return id_i_data



def purgueShortTrajs(fullDataset, min_time_len):
    totalDataset = []
    trajectoriesIDs = np.sort(np.unique(fullDataset['id']))
    for traj_id_i in range(0, len(trajectoriesIDs)):
        id_i = trajectoriesIDs[traj_id_i]
        id_i_data = fullDataset[fullDataset['id'] == id_i]
        dur_i = id_i_data.t.values[-1] - id_i_data.t.values[0]
        if dur_i >= min_time_len:
            if len(totalDataset) == 0:
                totalDataset = id_i_data.copy()
            else:
                totalDataset = totalDataset.append(id_i_data.copy())
    return totalDataset



def processTraj(code,n_digits,traj_id_i, n_traj, debug, show_time=10, save_time=600):
    '''
    Will check all the trajectories with id above the provided one [ID].
    Trajectories are read from files named:
        `./temp/traj_df-[ID].pickle

    Also does periodic savings on the vector file:
        './temp/dist_dict-[ID].pickle'

   dist_dict-[ID].pickle file contains 2 elements
        j: last trajectory we compared to
        dist_dict <id,dist>: keys are traj indexs, vals are distances

    '''
    # we add delta to dmin to distinguish crossing trajectories (delta) from non related (0) in sparse mat
    delta=1
    
    name = 'Traj-'+str(traj_id_i)
    if debug:
        print("[%s] Building dist row (%d)" % (name, traj_id_i))

    # get data for the base comparison    
    id_i_data = openIDFile(code,traj_id_i, n_digits)
    xi = id_i_data['x']
    yi = id_i_data['y']
    
    if debug:
        print("[%s] Processing (%d) points" % (name, len(id_i_data) ))

    min_t_i = id_i_data.index.min()
    max_t_i = id_i_data.index.max()

    # data pointers
    dist_dict = dict()
    curr_entry = traj_id_i + 1

    # check for temporal file ./temp/dist_dict-[ID].pickle
    temp_results_filename = getTempDistFileName(traj_id_i, n_digits,code)
    try:
        with open(temp_results_filename, 'rb') as fp:
            (curr_entry,dist_dict) = pickle.load(fp)
        if curr_entry == n_traj:
            print("[%s] Completed dict file with (%d) non zero entries. Exiting " % (name, len(dist_dict)))
            return
        else:
            print("[%s] Found temporal file at (%3.3f) \%" %
                  (name, 100.0 * (n_traj-curr_entry)/(n_traj-traj_id)))
    except IOError:
        if debug:
            print("[%s] Starting from scratch." % (name))

    last_show_time = time.time()
    last_save_time = time.time()
    for j in range(curr_entry, n_traj):
        if not isFIN():
            # get next trajectory to compare
            id_j_data = openIDFile(code,j, n_digits)
            min_t_j = id_j_data.index.min()
            max_t_j = id_j_data.index.max()

            intersect = not ((max_t_i) < (min_t_j)) or ((max_t_j) < (min_t_i))

            if intersect:
                xj = id_j_data['x']
                yj = id_j_data['y']
                
                dx = xj - xi
                dy = yj - yi
                d = np.sqrt(dx * dx + dy * dy)
                
                dmin = np.nanmin(d.values)
                if not np.isfinite(dmin):
                    print("[%s] NAN DISTANCE DETECTED to (%d)" % (name, j))
                dist_dict[j]=dmin+delta
                
            nowtime = time.time()
            if (nowtime - last_show_time) > show_time:
                last_show_time = time.time()
                print("[%s] Working on entry (%d)" % (name, j))
            if (nowtime - last_save_time) > save_time:
                last_save_time = time.time()
                print("[%s] Temp saving  at entry (%d)" % (name, j))
                saveDistDict(traj_id_i, n_digits,code, j, dist_dict)
        else:
            print("[%s] RECIVED EXIT SIGNAL!. Exiting " % (name, j))
            break
    if debug:
        print("[%s] Dict file is complete with (%d) non zero entries. Exiting " % (name, len(dist_dict)))
    saveDistDict(traj_id_i, n_digits,code, n_traj, dist_dict)
    
def setFIN():
    finFile = 'stop.txt'
    str = 'It is'
    np.savetxt(finFile, str, delimiter=',')

def unsetFIN():
    finFile = 'stop.txt'
    try:
        os.remove(finFile)
    except:
        pass

def isFIN():
    finFile = 'stop.txt'
    isFin = False
    try:
        fp = open(finFile, 'rb')
        isFin = True
        fp.close()
    except IOError:
        pass
    return isFin    

def getDataFrame(index, idList, globalDF):
    '''
    Returns a copy of a slice  from global dataframe
    indexed by time
    '''
    id_i = idList[index]
    id_i_data = globalDF[globalDF['id'] == id_i].copy()
    id_i_data.set_index('t', inplace=True)
    return id_i_data

def getPoint(df, index):
    try:
        p = df.iloc[index]
        point = np.array([p.x, p.y])
    except IndexError:
        print("POINT WITH INDEX %d DOES NOT EXIST" % index)
        point = None

    return point

def obtainQRSeq(df_1, df_2, QTC_ver):
    # df_1 = df_i
    # df_2 = df_j
    seq = []
    # we need a common timeline
    t1 = df_1.index
    t2 = df_2.index

    # time where the two trajs happened simultaneously:
    tmin=max(t1.min(),t2.min())
    tmax = min(t1.max(), t2.max())
    t = np.sort(np.unique(np.concatenate((t1, t2))))
    t=t[t>=tmin]
    t = t[t <= tmax]

    # df_1 = df_1.reindex(t)
    # df_1 = df_1.ffill()
    # df_1 = df_1.bfill()
    #
    # df_2 = df_2.reindex(t)
    # df_2 = df_2.ffill()
    # df_2 = df_2.bfill()
    # and now QSR
    for i in range(0, len(df_1.index.values)-1):
        p = getPoint(df_1, i)
        pn = getPoint(df_1, i+1)
        q = getPoint(df_2, i)
        qn = getPoint(df_2, i + 1)
        if QTC_ver == "QTCc":
            qsr_i = QTCc_pq(p, pn, q, qn)
        else:
            # print(len(t))
            qsr_i = QTCb_pq(p, pn, q, qn)

        seq.append(qsr_i)

    if len(seq)>0:
        if QTC_ver == "QTCc":
            qsr1, qsr2,qsr3, qsr4, dists = zip(*seq)
            seq = list(zip(qsr1, qsr2,qsr3, qsr4, dists, t[0:-1]))
        else:
            qsr1, qsr2, dists = zip(*seq)
            seq = list(zip(qsr1, qsr2, dists, t[0:-1]))

    else:
        print("No QSR interaction found!")
    return seq


def QTCb_pq(p, pn, q, qn):
    """
    QTC-B (QTC Basic) represents the 1D relative motion of these two points.
    It uses a 2-tuple of qualitative relations (t1,t2),
    where each element can assume any of the values {-, 0, +} as follows:

    -   t1 movement of p with respect to q:
            [-] p is moving towards q
            [0] p is stable with respect to q
            [+] p is moving away from q

        t1 can be represented by the sign of the cosine of the angle between P vector and QP vector, using dot product.
            P  is the vector formed by points p and pn
            QP is with q and p.

    -   t2 movement of q with respect to p: as above, but swapping p and q
    """
    proj_tol = 0.001
    norm_tol = 0.001

    if (p.ndim != pn.ndim) or (p.ndim != q.ndim) or (p.ndim > 1):
        print("Not all elements have dimension 1")
        return
    elif (p.shape != pn.shape) or (p.shape != q.shape):
        print("Not all elements have same num of components")
        return

    # vector pointing next position of p: P
    P = pn - p
    modP = LA.norm(P)

    # vector pointing next position of p: P
    Q = qn - q
    modQ = LA.norm(Q)

    # vector between p and reference q: QP
    QP = q - p

    # and oposite
    PQ = p - q

    # |PQ| == |QP|
    modQP = LA.norm(QP)

    # dot product of P and PQ
    dotP = P.dot(PQ)

    # dot product of Q and QP
    dotQ = Q.dot(QP)

    if modQP:
        # projection of vector P over QP vector
        p_over_q = dotP / modQP

        # projection of vector Q over PQ vector
        q_over_p = dotQ / modQP

#         # normal to vector P and QP vectors
#         p_normal_q = crosP / modQP

#         # normal to vector Q and PQ vectors
#         q_normal_p = crosQ / modQP
    else:
        # p and q are the same point ...
        p_over_q = 0
        q_over_p = 0

    t1 = getSign(p_over_q, proj_tol)
    t2 = getSign(q_over_p, proj_tol)

    return (t1, t2, modQP)


def QTCc_pq(p, pn, q, qn):
    """
    QTC Double-Cross represents the 2D relative motion of these two points.
    It uses a 4-tuple of qualitative relations (t1,t2,t3,t4),
    where each element can assume any of the values {-, 0, +} as follows:

    -   t1 movement of p with respect to q:
            [-] p is moving towards q
            [0] p is stable with respect to q
            [+] p is moving away from q

        t1 can be represented by the sign of the cosine of the angle between P vector and QP vector, using dot product.
            P  is the vector formed by points p and pn
            QP is with q and p.

    -   t2 movement of q with respect to p: as above, but swapping p and q

    -   t3 movement of p with respect to the connecting line between p and q (vector PQ):
            [-] p is moving to the left side of PQ
            [0] p is moving along PQ
            [+] p is moving to the right side of PQ

        t3 can be represented by the sign of the sine of the angle betwenn P vector and QP vector, using cross product.
            P  is the vector formed by points p and pn
            QP is with q and p.

     -   t4 movement of q with respect to the connecting line between q and p (vector QP) as above, but swapping p and q.

    """
    
#     proj_tol = 0.001
#     norm_tol = 0.001
    proj_tol = 0.001
    norm_tol = 0.001

    if (p.ndim != pn.ndim) or (p.ndim != q.ndim) or (p.ndim > 1):
        print("Not all elements have dimension 1")
        return
    elif (p.shape != pn.shape) or (p.shape != q.shape):
        print("Not all elements have same num of components")
        return

    # vector pointing next position of p: P
    P = pn - p
    modP = LA.norm(P)

    # vector pointing next position of p: P
    Q = qn - q
    modQ = LA.norm(Q)

    # vector between p and reference q: QP
    QP = q - p

    # and oposite
    PQ = p - q

    # |PQ| == |QP|
    modQP = LA.norm(QP)

    # dot product of P and PQ
    dotP = P.dot(PQ)

    # dot product of Q and QP
    dotQ = Q.dot(QP)

    # cross product of P and QP
    crosP = np.cross(P, QP)

    # cross product of Q and PQ
    crosQ = np.cross(Q, PQ)

    if modQP:
        # projection of vector P over QP vector
        p_over_q = dotP / modQP

        # projection of vector Q over PQ vector
        q_over_p = dotQ / modQP

        # normal to vector P and QP vectors
        p_normal_q = crosP / modQP

        # normal to vector Q and PQ vectors
        q_normal_p = crosQ / modQP
    else:
        # p and q are the same point ...
        p_over_q = 0
        q_over_p = 0
        p_normal_q = 0
        q_normal_p = 0

    t1 = getSign(p_over_q, proj_tol)
    t2 = getSign(q_over_p, proj_tol)
    t3 = getSign(p_normal_q, norm_tol)
    t4 = getSign(q_normal_p, norm_tol)

    return (t1, t2, t3,t4,modQP)


def getSign(val, tol):
    if (val > tol):
        signo = '+'
    elif (val < -tol):
        signo = '-'
    else:
        signo = '0'
    return signo


def loadRawData(code,min_trajectory_duration,colNames,dtypes,dataURI):
    allDataFilteredFileName = getAllDataFilteredFileName(code,min_trajectory_duration)

    try:
        allData = pd.read_csv(allDataFilteredFileName, names=colNames, dtype=dtypes)
        return allData
    except FileNotFoundError:
        print("Cant find File (%s) Filtering original data. Will take some minutes." % (allDataFilteredFileName))
        last_time = time.time()
        # load raw data
        allData = pd.read_csv(dataURI, names=colNames, dtype=dtypes)

        # drop -1 id trajectories
        allData = allData.drop(allData[allData['id'] == -1].index)

        # drop short trajectories
        allData = purgueShortTrajs(allData,min_trajectory_duration)

        # and save
        allData.to_csv(path_or_buf=allDataFilteredFileName,header=False)

        elapsed_time = time.time() - last_time
        print("Filtering dataframes lasted (%3.3f) " % str(timedelta(seconds=(elapsed_time))))
    return allData


def getTrajectoryIDs(code,allData):
    # load list with trajectory IDs
    trajectoriesIDFileName = getTrajectoriesFileName(code)

    try:
        trajectoriesIDs = np.genfromtxt(trajectoriesIDFileName, delimiter=',')
    except IOError:
        print("Cant find File (%s). Creating trajetories file." % (trajectoriesIDFileName))
        trajectoriesIDs = np.sort(np.unique(allData['id']))
        print("Saving trajectories file.")
        np.savetxt(trajectoriesIDFileName, trajectoriesIDs, delimiter=',')
    n_trajs = len(trajectoriesIDs)

    print("Dataframe has (%d) trajectories, (%d) entries " % (n_trajs, len(allData)))
    
    return (trajectoriesIDs,n_trajs)


def createQSR_seqs(code,min_trajectory_duration,colNames,dtypes,dataURI):
    print("Obtaining QSR relations between pairs" )  
    allData = loadRawData(code,min_trajectory_duration,colNames,dtypes,dataURI)

    last_time = time.time()  
    # (i,j)=list(pair_set)[0]
    QSR_seqs = []
    for i, j in pair_set:
        #print("(%d,%d)" % (i,j))
        dataframe_i = getDataFrame(i, trajectoriesIDs, allData)
        dataframe_j = getDataFrame(j, trajectoriesIDs, allData)
        QSR_seq = obtainQRSeq(dataframe_i, dataframe_j)
        QSR_seqs.append(QSR_seq)

    elapsed_time = time.time() - last_time
    print("This  took %s" % str(timedelta(seconds=(elapsed_time))))
    return QSR_seqs
    
def loadQSR_seqs(code,min_trajectory_duration,colNames,dtypes,dataURI):
    QSR_seqsFileName = getResultsFolder(code)+'QSR_seqs.pickle'
    try:
        # load QSR interactions
        fp = open(getResultsFolder(code)+'QSR_seqs.pickle', 'rb')
        QSR_seqs = pickle.load(fp)        
    except IOError:
        QSR_seqs = createQSR_seqs(code,min_trajectory_duration,colNames,dtypes,dataURI)
        print("Saving QSR relations between pairs" )  
        pickle.dump(QSR_seqs, open(QSR_seqsFileName, "wb"))
        
    return QSR_seqs

def extractStateSeqsVec(Q):
    X = []
    lengths=[]
    for i in range(0,len(Q)):
        # get a sequence
        qsr_seq = Q[i]
        # if contains data
        if len(qsr_seq)>0:
            # unzip distance and normal components
            qsr1, qsr2, qsr3, qsr4, dists,t = zip(*qsr_seq )
            # encode states into nums. Python rules!
            qInt_l = [np.array([stateCharToInt(q1), 
                                stateCharToInt(q2), 
                                stateCharToInt(q3), 
                                stateCharToInt(q4)]) for q1, q2, q3, q4 in zip(qsr1, qsr2,qsr3, qsr4)]
            qInt = np.array(qInt_l)
            if X==[]:
                X = qInt
            else:
                X= np.concatenate((X,qInt))
            # we need sequence length for HMM
            lengths.append(len(qInt))

    # X must be n_samples,n_features
    # lengths is n_sequences, but lengths.sum()==n_sequences
    return (X,lengths)


def stateCharToInt(stChar):
    states={'+':2,
            '0': 1,
            '-': 0}
    return states[stChar]


def stateIntToChar(stInt):
    '''
    Used to cast state number to QTC char
    '''  
    states={2:'+',
            1:'0',
            0:'-'}
    return states[stInt]

def decodeState(codeDec,nbits=4):
    #nbits = int(np.ceil(np.log(codeDec) / np.log(3)))
    d = []
    tmp = codeDec
    for i in range(0,nbits-1):
        di = int(tmp) % 3
        tmp = tmp // 3
        d.append(di)
    d.append(tmp%3)
    d.reverse()
    state = [stateIntToChar(di) for di in d]

    return state