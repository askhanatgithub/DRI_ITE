from torch import nn   
import torch
class Tclassifier(nn.Module):
    def __init__(self,
                 input_dim,
                 regularization):
        super(Tclassifier, self).__init__()
       
        
        #Classifier to calculate weights
        self.classifier_w1 = nn.Sequential(
            nn.Linear(input_dim, 100),
            
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.classifier_w2 = nn.Sequential(
            nn.Linear(100, 100),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
       
        self.classifier_w3 = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()
        
        


    def forward(self, inputs):
       
        
        # classifires
       
        out_w=self.classifier_w1(inputs)
        out_w=self.classifier_w2(out_w)
        
       
        out_w_f=self.sig(self.classifier_w3(out_w))
        
        
        # Returning arguments

       
        return out_w_f
    
class Regressors(nn.Module):
    def __init__(self,
                 input_dim,hid_dim,
                 regularization):
        super(Regressors, self).__init__()
        
    

        self.regressor1_y0 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor2_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor3_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
            
        self.regressor4_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor5_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
                
        self.regressor6_y0 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
          
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
    
        self.regressorO_y0 = nn.Linear(hid_dim, 1)
        
        

        self.regressor1_y1 = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        self.regressor2_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor3_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor4_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor5_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
        self.regressor6_y1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
           
            nn.ELU(),
            nn.Dropout(p=regularization),
        )
        
       
        self.regressorO_y1 = nn.Linear(hid_dim, 1)
        
        


    def forward(self, inputs):

        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        out_y0 = self.regressor1_y0(inputs)
        out_y0 = self.regressor2_y0(out_y0)
        out_y0 = self.regressor3_y0(out_y0)
        #out_y0 = self.regressor4_y0(out_y0)  
        #out_y0 = self.regressor5_y0(out_y0)
        #out_y0 = self.regressor6_y0(out_y0)
        y0 = self.regressorO_y0(out_y0)

        out_y1 = self.regressor1_y1(inputs)
        out_y1 = self.regressor2_y1(out_y1)
        out_y1 = self.regressor3_y1(out_y1)
        #out_y1 = self.regressor4_y1(out_y1)
        #out_y1 = self.regressor5_y1(out_y1)
        #out_y1 = self.regressor6_y1(out_y1)
        
        y1 = self.regressorO_y1(out_y1)
        
        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))
        
        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))
        
        # Returning arguments

        concat = torch.cat((y0, y1), 1)
        return concat#out_w_f,out_t_f


    
class Decoder(nn.Module):
    def __init__(self, input_dim, decoding_dim,regularization):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 600),  # First decoder layer, 200 dimensions
            nn.ELU(),
            nn.Dropout(p=regularization),
           
            nn.Linear(600, 400),  # First decoder layer, 200 dimensions
            nn.ELU(),
            nn.Dropout(p=regularization),
            

            

            nn.Linear(400, decoding_dim)    # Third decoder layer, output original dimensions
            
            
        )
        self.sig=nn.Sigmoid()

    def forward(self, x):
        decoded = self.decoder(x)
        decoded_a=decoded[:,0:6]
        decoded_b=self.sig(decoded[:,6:25])
        decoded_c=decoded[:,25:]
        return torch.cat((decoded_a, decoded_b,decoded_c), 1)

class TarNet(nn.Module):
    def __init__(self,
                 input_dim,lat_dim_enc,
                 regularization):
        super(TarNet, self).__init__()
        self.encoder_gamma_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_gamma_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_gamma_3 = nn.Linear(lat_dim_enc, lat_dim_enc)
        
        self.encoder_delta_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_delta_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_delta_3 = nn.Linear(lat_dim_enc, lat_dim_enc)
        
        self.encoder_upsilon_1 = nn.Linear(input_dim,lat_dim_enc)
        self.encoder_upsilon_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.encoder_upsilon_3 = nn.Linear(lat_dim_enc, lat_dim_enc)
        
        self.Irr_1 = nn.Linear(input_dim,lat_dim_enc)
        self.Irr_2 = nn.Linear(lat_dim_enc, lat_dim_enc)
        self.Irr_3 = nn.Linear(lat_dim_enc, lat_dim_enc)
        
        self.sig = nn.Sigmoid()
        self.BN= nn.BatchNorm1d(lat_dim_enc)
    

      
        


    def forward(self, inputs):
        x_gamma = nn.functional.elu(self.encoder_gamma_1(inputs))
        x_gamma = nn.functional.elu(self.encoder_gamma_2(x_gamma))
        phi_gamma = self.encoder_gamma_3(x_gamma)
        
        x_delta = nn.functional.elu(self.encoder_delta_1(inputs))
        x_delta = nn.functional.elu(self.encoder_delta_2(x_delta))
        phi_delta = self.encoder_delta_3(x_delta)
        
        x_upsilon = nn.functional.elu(self.encoder_upsilon_1(inputs))
        x_upsilon = nn.functional.elu(self.encoder_upsilon_2(x_upsilon))
        phi_upsilon = self.encoder_upsilon_3(x_upsilon)

        
        x_irr = nn.functional.elu(self.Irr_1(inputs))
        x_irr = nn.functional.elu(self.Irr_2(x_irr))
        phi_irr = self.Irr_3(x_irr)
        
        # Regressors
        #del_ups=torch.cat((phi_delta, phi_upsilon), 1)
        #out_y0 = self.regressor1_y0(del_ups)
        #out_y0 = self.regressor2_y0(out_y0)
        #y0 = self.regressorO_y0(out_y0)

        #out_y1 = self.regressor1_y1(del_ups)
        #out_y1 = self.regressor2_y1(out_y1)
        #y1 = self.regressorO_y1(out_y1)
        
        # classifires
        #gam_del=torch.cat((phi_gamma,phi_delta), 1)
        #out_w=self.classifier_w1(phi_delta)
        #out_w=self.classifier_w2(out_w)
        #out_w_f=self.sig(self.classifier_w3(out_w))
        
        #out_t=self.classifier_t1(gam_del)
        #out_t=self.classifier_t2(out_t)
        #out_t_f=self.sig(self.classifier_t3(out_t))
        
        # Returning arguments

        #concat = torch.cat((y0, y1), 1)
        return phi_gamma,phi_delta,phi_upsilon,phi_irr#out_w_f,out_t_f
