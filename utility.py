from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    

def get_dataloader(x_data,y_data,batch_size):

    x_train_sr=x_data[x_data['treatment']==0]
    y_train_sr=y_data[x_data['treatment']==0]
    x_train_tr=x_data[x_data['treatment']==1]
    y_train_tr=y_data[x_data['treatment']==1]


    train_data_sr = Data(np.array(x_train_sr), np.array(y_train_sr))
    train_dataloader_sr = DataLoader(dataset=train_data_sr, batch_size=batch_size)

    train_data_tr = Data(np.array(x_train_tr), np.array(y_train_tr))
    train_dataloader_tr = DataLoader(dataset=train_data_tr, batch_size=batch_size)


    return train_dataloader_sr, train_dataloader_tr

def get_data(data_type,file_num):
   

    if(data_type=='train'):
        data=pd.read_csv(f"Data_IHDP/ihdp_npci_train_{file_num}.csv")
    else:
        data = pd.read_csv(f"Data_IHDP/ihdp_npci_test_{file_num}.csv")

    x_data=pd.concat([data.iloc[:,0], data.iloc[:, 1:30]], axis = 1)
    x_data.iloc[:,18]=np.where(x_data.iloc[:,18]==2,1,0)
    #x_data_a=x_data.iloc[:,0:5]
    #x_data_b=x_data.iloc[:,5:30]
    #scaler.fit(x_data_b)
    #scaled_b = pd.DataFrame(scaler.fit_transform(x_data_b))
    #x_data=data.iloc[:, 5:30]
    #x_data_trans=pd.concat([x_data_a,scaled_b],axis=1)
    y_data_trans=data.iloc[:, 1]
    #y_data_trans=pd.DataFrame(scaler.fit_transform(data.iloc[:, 1].to_numpy().reshape(-1, 1)))
    #y_data_trans=y_data_trans.to_numpy().reshape(-1,)
    return x_data,y_data_trans

def regression_loss(concat_true, concat_pred):
    #computes a standard MSE loss for TARNet
    y_true = concat_true[:, 0] #get individual vectors
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    #Each head outputs a prediction for both potential outcomes
    #We use t_true as a switch to only calculate the factual loss
    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    #note Shi uses tf.reduce_sum for her losses instead of tf.reduce_mean.
    #They should be equivalent but it's possible that having larger gradients accelerates convergence.
    #You can always try changing it!
    return loss0 + loss1
def calc_mmdsq(Phic, Phit):
    #Phic, Phit = torch.split(Phi, [torch.sum(t == 0), torch.sum(t == 1)], dim=0)
    
    Phic=Phic
    Phit=Phit
    sig=0.1
    #p=1e-4
    p=0.5
    sig = torch.tensor(sig)
    Kcc = torch.exp(-torch.cdist(Phic, Phic, 2.0001) / torch.sqrt(sig))
    Kct = torch.exp(-torch.cdist(Phic, Phit, 2.0001) / torch.sqrt(sig))
    Ktt = torch.exp(-torch.cdist(Phit, Phit, 2.0001) / torch.sqrt(sig))

    m = Phic.shape[0]
    n = Phit.shape[0]

    mmd = (1 - p) ** 2 / (m * m) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * n) * (Ktt.sum() - n)
    mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4
    '''
    mmd = (1 - p) ** 2 / (m * (m - 1)) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * (n - 1)) * (Ktt.sum() - n)
    mmd -= - 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4
    '''
    return mmd

def mmdsq_loss(concat_true,concat_pred):
    t_true = concat_true[:, 1]
    p=split_pred(concat_pred)
    mmdsq_loss = torch.mean(calc_mmdsq(p['phi'],t_true))
    return mmdsq_loss

def split_pred(concat_pred):
      #generic helper to make sure we dont make mistakes
    preds={}
    preds['y0_pred'] = concat_pred[:, 0]
    preds['y1_pred'] = concat_pred[:, 1]
    preds['phi'] = concat_pred[:, 2:]
    return preds

def calc_wasse(Phi0, phi1):
    
    loss1 = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    #num_zero=torch.zeros((Phi.shape[0]),1)
    #num_ones=torch.ones((Phi.shape[0]),1)
    #phi_zero=torch.cat((Phi,num_zero),1)
    #phi_ones=torch.cat((Phi,num_ones),1)
    
    wasser = loss1(Phi0, phi1)
 
    return wasser


def cfr_loss(concat_true,concat_pred,upsilon):
    alpha=1
    lossR = regression_loss(concat_true,concat_pred)
    #lossIPM = mmdsq_loss(concat_true,concat_pred)
    #change wasserstein loss function accordingly
    wass_loss=calc_wasse()
    return lossR,wass_loss

def weight_cal(true_t,sigmoid_prob,prop_t1):
    
    prop_t0=1-prop_t1
    weight_10=(prop_t0/(1-prop_t0))
    weight_11=(prop_t1/(1-prop_t1))
   
    w_t = (1/(2.*prop_t1))
    w_c = (1/(2.*(1.-prop_t1))) 
    weights=torch.where(true_t== 0, (1+(weight_10*((1-sigmoid_prob)/sigmoid_prob))*(w_t+w_c)),(1+(weight_11*((1-sigmoid_prob)/sigmoid_prob)))*(w_t+w_c))                       
    #weights=torch.where(true_t== 0,w_c,w_t)
    return weights

def cal_pehe(data,y,Regressor,Encoder):
    #data,y=get_data('test',i)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data=data.to_numpy()
    data=torch.from_numpy(data.astype(np.float32)).to(device)
    #Replace 30 with :

    phi_gamma,phi_delta,phi_upsilon,phi_irr=Encoder(data[:,5:])
    del_ups=torch.cat((phi_delta, phi_upsilon), 1)
    
   
    #del_ome=torch.cat((phi_delta,phi_irr), 1)
    #ups_ome=torch.cat((phi_upsilon,phi_irr), 1)
    
    # change to del_ups for true evaluation
    concat_pred=Regressor(del_ups)
    
    
    
    t=data[:,0]
   
    
    
    
    predicted_y=torch.where(t.squeeze() == 0, concat_pred[:,0], concat_pred[:,1])
    
    #print(y)
    #print('mae test',np.mean(np.abs(predicted_y.detach().numpy()-y)))
    
    #print('mae train',np.mean(np.abs(predicted_yt.detach().numpy()-ty)))
    
    #concat_num=scaler.inverse_transform(pd.DataFrame(concat.detach().numpy() ))
    #concat_pred=torch.from_numpy(concat_num.astype(np.float32))
    #dont forget to rescale the outcome before estimation!
    #y0_pred = data['y_scaler'].inverse_transform(concat_pred[:, 0].reshape(-1, 1))
    #y1_pred = data['y_scaler'].inverse_transform(concat_pred[:, 1].reshape(-1, 1))
    cate_pred=concat_pred[:,1]-concat_pred[:,0]
    cate_true=data[:,4]-data[:,3] #Hill's noiseless true values


    cate_err=torch.mean( torch.square( ( (cate_true) - (cate_pred) ) ) )

    return torch.sqrt(cate_err).item()

def cal_pehe_nn(data,y,Reg,Enc):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        datat=data.to_numpy()
        datat=torch.from_numpy(datat.astype(np.float32)).to(device)
        yt=y.to_numpy()
        yt=torch.from_numpy(yt.astype(np.float32)).to(device)
        df_datac=data[data['treatment']==0]
        df_datat=data[data['treatment']==1]
        
        
        torch_c=df_datac.to_numpy()
        torch_c=torch.from_numpy(torch_c.astype(np.float32))
        torch_t=df_datat.to_numpy()
        torch_t=torch.from_numpy(torch_t.astype(np.float32))
        
        
        
        
        
        phi_gamma,phi_delta,phi_upsilon,phi_irr=Enc(datat[:,5:])
        del_ups=torch.cat((phi_delta, phi_upsilon), 1)

        concat_pred=Reg(del_ups)
        
        
       
        dists = torch.sqrt(torch.cdist(torch_c, torch_t))
        
        c_index=torch.argmin(dists, dim=0).tolist()
        t_index=torch.argmin(dists, dim=1).tolist()
    
        yT_nn=df_datac.iloc[c_index]['y_factual']
        yC_nn=df_datat.iloc[t_index]['y_factual']
        yT_nn=yT_nn.to_numpy()
        yT_nn=torch.from_numpy(yT_nn.astype(np.float32)).to(device)
        yC_nn=yC_nn.to_numpy()
        yC_nn=torch.from_numpy(yC_nn.astype(np.float32)).to(device)
        y_nn = torch.cat([yT_nn, yC_nn],0) 
        

        
      
        cate_pred=concat_pred[:,1]-concat_pred[:,0]

        
        cate_nn_err=torch.mean(torch.square((((1 - 2 * datat[:,0]) * (y_nn - yt)) - cate_pred)))
        return cate_nn_err.item()
        #torch.mean( torch.square( (1-2*datat[:,0]) * (y_nn-y) - (concat_pred[:,1]-concat_pred[:,0]) ) )
    
        


def wasserstein(X, t, p=0.5, lam=10, its=10, sq=False, backpropT=False):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    it = torch.where(t > 0)[0]
    ic = torch.where(t < 1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = torch.tensor(Xc.shape[0], dtype=torch.float)
    nt = torch.tensor(Xt.shape[0], dtype=torch.float)

    if sq:
        M = torch.cdist(Xt, Xc, p=2) ** 2
    else:
        M = torch.sqrt(torch.cdist(Xt, Xc, p=2) ** 2)

    M_mean = torch.mean(M)
    M_drop = torch.nn.functional.dropout(M, p=10 / (nc * nt))
    delta = torch.max(M).detach()
    eff_lam = lam / M_mean

    num_row = M.shape[0]
    num_col = M.shape[1]
    row = delta * torch.ones(1, num_col).to(device)
    col = torch.cat([delta * torch.ones(num_row, 1).to(device), torch.zeros(1, 1).to(device)], dim=0)
    Mt = torch.cat([M, row], dim=0)
    Mt = torch.cat([Mt, col], dim=1)

    a = torch.cat([p * torch.ones(len(it), 1) / nt, (1 - p) * torch.ones(1, 1)], dim=0).to(device)
    b = torch.cat([(1 - p) * torch.ones(len(ic), 1) / nc, p * torch.ones(1, 1)], dim=0).to(device)

    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam).to(device) + 1e-6
    U = K * Mt
    ainvK = K / a

    u = a.clone()
    for i in range(its):
        u = 1.0 / (ainvK @ (b / (u.T @ K).T))
    v = b / (u.T @ K).T

    T = u * (v.T * K)

    if not backpropT:
        T = T.detach()

    E = T * Mt
    D = 2 * torch.sum(E)

    return D, Mlam
  
def cal_weightmatr(Encoder):
    layer=Encoder.children()
    layer_list=list(layer)
    inter_WG=torch.matmul(layer_list[0].weight.t(),layer_list[1].weight.t())
    WG=torch.matmul(inter_WG,layer_list[2].weight.t())
    #len(list(layer))

    inter_WD=torch.matmul(layer_list[3].weight.t(),layer_list[4].weight.t())
    WD=torch.matmul(inter_WD,layer_list[5].weight.t())

    inter_WU=torch.matmul(layer_list[6].weight.t(),layer_list[7].weight.t())
    WU=torch.matmul(inter_WU,layer_list[8].weight.t())
    
    inter_Irr=torch.matmul(layer_list[9].weight.t(),layer_list[10].weight.t())
    IU=torch.matmul(inter_Irr,layer_list[11].weight.t())


    return WG,WD,WU,IU

def cal_RO(WM1,WM2,WM3,WM4):
    WM1_bar=torch.mean(torch.abs(WM1),dim=-1)
    WM2_bar=torch.mean(torch.abs(WM2),dim=-1)
    WM3_bar=torch.mean(torch.abs(WM3),dim=-1)
    WM4_bar=torch.mean(torch.abs(WM4),dim=-1)
    """
    RO_1=torch.dot(WM1_bar.t(),WM2_bar)
    RO_2=torch.dot(WM2_bar.t(),WM3_bar)
    RO_3=torch.dot(WM3_bar.t(),WM1_bar)
    """
    RO_1=torch.inner(WM1_bar,WM2_bar)
    RO_2=torch.inner(WM2_bar,WM3_bar)
    RO_3=torch.inner(WM3_bar,WM1_bar)
    
    
    """
    ROI_1=torch.dot(WM4_bar.t(),WM1_bar)
    ROI_2=torch.dot(WM4_bar.t(),WM2_bar)
    ROI_3=torch.dot(WM4_bar.t(),WM3_bar)
    
    """
    ROI_1=torch.inner(WM4_bar,WM1_bar)
    ROI_2=torch.inner(WM4_bar,WM2_bar)
    ROI_3=torch.inner(WM4_bar,WM3_bar)
    OR=(RO_1+RO_2+RO_3+ROI_1+ROI_2+ROI_3)
    
    return OR

def OR_reg(WM1,WM2,WM3,WM4):
    WM1_bar=torch.mean(torch.abs(WM1),dim=-1)
    WM2_bar=torch.mean(torch.abs(WM2),dim=-1)
    WM3_bar=torch.mean(torch.abs(WM3),dim=-1)
    WM4_bar=torch.mean(torch.abs(WM4),dim=-1)
    
    a=torch.square(torch.sum(WM1_bar)-1)
    b=torch.square(torch.sum(WM2_bar)-1)
    c=torch.square(torch.sum(WM3_bar)-1)
    d=torch.square(torch.sum(WM4_bar)-1)
    
    return a+b+c+d

def loss_cal(X_data,y_data, Embedder,Regres, Tclass):
    
    x_train_sr=X_data[X_data['treatment']==0]
    y_train_sr=y_data[X_data['treatment']==0]
    x_train_tr=X_data[X_data['treatment']==1]
    y_train_tr=y_data[X_data['treatment']==1]
    xs_t=x_train_sr.iloc[:,0].to_numpy()
    xt_t=x_train_tr.iloc[:,0].to_numpy()
    #Replace 30 with :
    xs=x_train_sr.iloc[:,5:].to_numpy()
    xt=x_train_tr.iloc[:,5:].to_numpy()
    xs_tt=torch.from_numpy(xs_t.astype(np.float32))
    xt_tt=torch.from_numpy(xt_t.astype(np.float32))
    y_train_sr=y_train_sr.to_numpy()
    y_train_tr=y_train_tr.to_numpy()
    xs=torch.from_numpy(xs.astype(np.float32))
    xt=torch.from_numpy(xt.astype(np.float32))
    
    y_train_sr=torch.from_numpy(y_train_sr.astype(np.float32))
    y_train_tr=torch.from_numpy(y_train_tr.astype(np.float32))
    
    
    input_data=torch.cat((xs,xt),0)
    true_y=torch.unsqueeze(torch.cat((y_train_sr,y_train_tr),0), dim=1)
    true_t=torch.unsqueeze(torch.cat((xs_tt,xt_tt),0), dim=1)
    
    # Calculate loss here and return
    prop_t1=(xt.shape[0]/input_data.shape[0])
            
            
    phi_gamma,phi_delta,phi_upsilon,Irr=Embedder(input_data)
    del_ups=torch.cat((phi_delta, phi_upsilon), 1)
    gam_del=torch.cat((phi_gamma,phi_delta), 1)
    concat_pred=Regres(del_ups)
    #w_f=Wclass(phi_delta)
    w_t=Tclass(gam_del)

    predicted_y=torch.unsqueeze(torch.where(true_t.squeeze() == 0, concat_pred[:,0], concat_pred[:,1]),dim=1)
 
    #weights=weight_cal(true_t,w_f,prop_t1)

    #Rloss=regression_loss(concat_true,concat_pred)
    Rloss=MSE(predicted_y,true_y)
    #dot=torch.dot(weights.squeeze(),Rloss.squeeze())
    #Rlossmean=(dot/input_data.shape[0])
    Wassloss,dist=wasserstein(phi_upsilon,true_t)
    #mmd=calc_mmdsq(phi_upsilon[0:xs_train.shape[0],:],phi_upsilon[xs_train.shape[0]:,:])
    #print('val')
    Tloss=BCE(w_t, true_t)



    #wclassloss=BCE(w_f, true_t)
    
    w1,w2,w3,w4=cal_weightmatr(Embedder)
    OR=cal_RO(w1,w2,w3,w4)
    OR_re=OR_reg(w1,w2,w3,w4)
    #combined loss
    combined_loss=Rloss+(100*Wassloss)+(100*Tloss)+(OR)+(200*OR_re)
            
    
    
    return combined_loss.item()

def add_dummy_features(data, no_features):
    np.random.seed(1)
    data_dummy=pd.DataFrame()
    dummies=no_features

    if(dummies==0):
        return data
    
    elif (dummies==1):
        data_aux = pd.DataFrame({f"d{0}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
        data_dummy = pd.concat([data_dummy, data_aux],axis=1)
    else:
        for i in range(dummies):
            data_aux = pd.DataFrame({f"d{i}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
            data_dummy = pd.concat([data_dummy, data_aux],axis=1)
            
    new_data=pd.concat([data,data_dummy],axis=1)   

    return new_data


def add_dummy_features_shuffle(data, no_features):
    np.random.seed(2)
    data_dummy=pd.DataFrame()
    dummies=no_features

    if(dummies==0):
        return data
    
    elif (dummies==1):
        #data_aux = pd.DataFrame({f"d{0}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
        #arr=data.iloc[:,5].values
        #data_aux = pd.DataFrame({f"d{0}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
        arr=data.iloc[:,5].values
        np.random.shuffle(arr)
        data_aux = pd.DataFrame({f"d{0}"  :arr.tolist() })
        #data_aux = pd.DataFrame({f"d{0}"  :np.random.shuffle(arr)},index=[0])
        data_dummy = pd.concat([data_dummy, data_aux],axis=1)
    else:
        for i in range(5,5+dummies):
            #np.random.seed(i)
            #data_aux = pd.DataFrame({f"d{i}"  : np.random.randint(low=0, high=2, size=data.shape[0])})
            #data_aux = pd.DataFrame({f"d{i}"  : data.iloc[:,5].sample(frac=1).reset_index(drop=True)})
            arr=data.iloc[:,5].values
            np.random.shuffle(arr)
            data_aux = pd.DataFrame({f"d{i}"  :arr.tolist() })
            data_dummy = pd.concat([data_dummy, data_aux],axis=1)
            
    new_data=pd.concat([data,data_dummy],axis=1)   

    return new_data