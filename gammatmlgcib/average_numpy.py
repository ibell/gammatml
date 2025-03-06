import numpy as np
import pandas as pd
import os

import ctREFPROP.ctREFPROP as ct

###### Functions to run the model
def descriptors_scaling(input_i, input_j, temperature, descriptors_list):
    """
    Scale the descriptors of two inputs and the temperature using mean and standard deviation values.

    Parameters:
    input_i (dict): Dictionary containing the descriptors of input i.
    input_j (dict): Dictionary containing the descriptors of input j.
    temperature (float): Temperature value.
    descriptors_list (list): List of descriptors to be scaled.

    Returns:
    tuple: A tuple containing the scaled descriptors of input i, input j, and the scaled temperature.

    Example:
    input_i = {'TC [K]': 400, 'PC [Pa]': 5000000}
    input_j = {'TC [K]': 380, 'PC [Pa]': 4500000}
    descriptors_list = ['TC [K]', 'PC [Pa]']

    scaled_input_i, scaled_input_j = descriptors_scaling(input_i, input_j, descriptors_list)
    print(scaled_input_i)  # Output: [0.0, -0.5]
    print(scaled_input_j)  # Output: [-0.5, -1.0]

    Possible descriptors list:
    - ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']
    - ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]', 'TTRP [K]']
    - ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]', 'TTRP [K]', 'PTRP [Pa]']
    - ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]', 'TTRP [K]', 'PTRP [Pa]', 'MM [kg/mol]']
    """

    # These values are required to standardize the descriptors
    # x' = (x - x.mean()) / x.std()
    scaler_mean_dict = {'T/K': 309.40173315858993,
                        'TC [K]': 373.08515837763343,
                        'PC [Pa]': 4520786.299862178,
                        'ACF [-]': 0.24087281748375663,
                        'DIPOLE [debye]': 1.0095449852333136,
                        'TTRP [K]': 144.47859746012995,
                        'PTRP [Pa]': 39476.57255022334,
                        'MM [kg/mol]': 0.08353055596928527}

    scaler_std_dict =  {'T/K': 48.75102381063749,
                        'TC [K]': 49.514905589182206,
                        'PC [Pa]': 1479473.3020734116,
                        'ACF [-]': 0.06453683727455967,
                        'DIPOLE [debye]': 0.8988423794801139,
                        'TTRP [K]': 39.34477499922218,
                        'PTRP [Pa]': 135553.50420303186,
                        'MM [kg/mol]': 0.04505098247583611}

    # inputs are dictionaries in the form {'T': ..., 'TC [K]': ... }
    scaled_input_i = input_i.copy()
    scaled_input_j = input_j.copy()

    scaled_temperature = (temperature - scaler_mean_dict['T/K']) / scaler_std_dict['T/K']

    for key in descriptors_list:
        scaled_input_i[key] = (scaled_input_i[key] - scaler_mean_dict[key]) / scaler_std_dict[key]
        scaled_input_j[key] = (scaled_input_j[key] - scaler_mean_dict[key]) / scaler_std_dict[key]

    # order of the descriptors is maintained
    # descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]', 'TTRP [K]', 'PTRP [Pa]', 'MM [kg/mol]']
    scaled_input_i = list(scaled_input_i.values())
    scaled_input_j = list(scaled_input_j.values())
    return scaled_input_i, scaled_input_j, scaled_temperature


def numpy_AverageModel_Tind(scaled_input_i, scaled_input_j, scaled_temperature, ann_params, check_similarity=False):
    """
    Calculates the average of a predicted value using a neural network model.

    Parameters:
    scaled_input_i (numpy.ndarray): The scaled input for the first input vector.
    scaled_input_j (numpy.ndarray): The scaled input for the second input vector.
    scaled_temperature (float): The scaled temperature value.
    ann_params (dict): A dictionary containing the parameters of the neural network model.
    check_similarity (bool): If True, the similarity between the two inputs is considered. Default is False.

    The scaled inputs are obtained using the descriptors_scaling function.

    Returns:
    float: The average of evaluting with [scaled_input_i, scaled_input_j]
    and [scaled_input_j, scaled_input_i] two inputs.

    """

    # Function implementation...

    # ANN inputs
    ann_i = np.hstack([scaled_input_i, scaled_input_j])
    ann_j = np.hstack([scaled_input_j, scaled_input_i])

    # evaluating the hidden layers
    n_hidden_layers = ann_params['n_hidden_layers']
    for i in range(n_hidden_layers):
        ann_i = np.matmul(ann_i, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_i = np.tanh(ann_i)

        ann_j = np.matmul(ann_j, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_j = np.tanh(ann_j)

    # evaluating the output Layer
    ann_i = np.matmul(ann_i, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']
    ann_j = np.matmul(ann_j, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']

    # obtaining the gammaT value
    gammaT_ann = (ann_i + ann_j) / 2.
    gammaT_ann = float(gammaT_ann)

    if check_similarity:
        norm_i = np.linalg.norm(scaled_input_i)
        norm_j = np.linalg.norm(scaled_input_j)
        cos_thetha = np.dot(scaled_input_i, scaled_input_j) / (norm_i * norm_j)
        gammaT_ann = 1. + gammaT_ann * (1. - cos_thetha)

    return gammaT_ann


def numpy_AverageModel_Tdep(scaled_input_i, scaled_input_j, scaled_temperature, ann_params, check_similarity=False):
    """
    Calculates the average of a predicted value using a neural network model.

    Parameters:
    scaled_input_i (numpy.ndarray): The scaled input for the first input vector.
    scaled_input_j (numpy.ndarray): The scaled input for the second input vector.
    scaled_temperature (float): The scaled temperature value.
    ann_params (dict): A dictionary containing the parameters of the neural network model.
    check_similarity (bool): If True, the similarity between the two inputs is considered. Default is False.

    The scaled inputs are obtained using the descriptors_scaling function.

    Returns:
    float: The average of evaluting with [scaled_input_i, scaled_input_j]
    and [scaled_input_j, scaled_input_i] two inputs.

    """

    # Function implementation...

    # ANN inputs
    ann_i = np.hstack([scaled_input_i, scaled_input_j, scaled_temperature])
    ann_j = np.hstack([scaled_input_j, scaled_input_i, scaled_temperature])

    # evaluating the hidden layers
    n_hidden_layers = ann_params['n_hidden_layers']
    for i in range(n_hidden_layers):
        ann_i = np.matmul(ann_i, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_i = np.tanh(ann_i)

        ann_j = np.matmul(ann_j, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_j = np.tanh(ann_j)

    # evaluating the output Layer
    ann_i = np.matmul(ann_i, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']
    ann_j = np.matmul(ann_j, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']

    # obtaining the gammaT value
    gammaT_ann = (ann_i + ann_j) / 2.
    gammaT_ann = float(gammaT_ann)

    if check_similarity:
        norm_i = np.linalg.norm(scaled_input_i)
        norm_j = np.linalg.norm(scaled_input_j)
        cos_thetha = np.dot(scaled_input_i, scaled_input_j) / (norm_i * norm_j)
        gammaT_ann = 1. + gammaT_ann * (1. - cos_thetha)

    return gammaT_ann


def numpy_AverageModel_linear(scaled_input_i, scaled_input_j, scaled_temperature, ann_params, check_similarity=False):
    """
    Calculates the average of a predicted value using a neural network model.

    Parameters:
    scaled_input_i (numpy.ndarray): The scaled input for the first input vector.
    scaled_input_j (numpy.ndarray): The scaled input for the second input vector.
    scaled_temperature (float): The scaled temperature value.
    ann_params (dict): A dictionary containing the parameters of the neural network model.
    check_similarity (bool): If True, the similarity between the two inputs is considered. Default is False.

    The scaled inputs are obtained using the descriptors_scaling function.

    Returns:
    float: The average of evaluting with [scaled_input_i, scaled_input_j]
    and [scaled_input_j, scaled_input_i] two inputs.

    """

    # Function implementation...

    # ANN inputs
    ann_i = np.hstack([scaled_input_i, scaled_input_j])
    ann_j = np.hstack([scaled_input_j, scaled_input_i])

    # evaluating the hidden layers
    n_hidden_layers = ann_params['n_hidden_layers']
    for i in range(n_hidden_layers):
        ann_i = np.matmul(ann_i, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_i = np.tanh(ann_i)

        ann_j = np.matmul(ann_j, ann_params[f'layer_{i}_kernel']) + ann_params[f'layer_{i}_bias']
        ann_j = np.tanh(ann_j)

    # evaluating the output Layer
    ann_i = np.matmul(ann_i, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']
    ann_j = np.matmul(ann_j, ann_params['output_layer_kernel']) + ann_params['output_layer_bias']

    m_b = (ann_i + ann_j) / 2
    m_b = np.atleast_2d(m_b)
    gammaT_ann = m_b[:, 0] + m_b[:, 1] * scaled_temperature
    gammaT_ann = float(gammaT_ann)

    if check_similarity:
        norm_i = np.linalg.norm(scaled_input_i)
        norm_j = np.linalg.norm(scaled_input_j)
        cos_thetha = np.dot(scaled_input_i, scaled_input_j) / (norm_i * norm_j)
        gammaT_ann = 1. + gammaT_ann * (1. - cos_thetha)

    return gammaT_ann


####

class GammaCalculator:
    def __init__(self, folder_to_read, path_descriptors, descriptors_list, model_type='Tdep', 
                 check_similarity=False):

        if model_type == 'Tdep':
            self.model = numpy_AverageModel_Tdep
        elif model_type == 'Tind':
            self.model = numpy_AverageModel_Tind
        elif model_type == 'linear':
            self.model = numpy_AverageModel_linear
        else:
            raise ValueError('Model type not recognized. Choose between Tdep, Tind, linear.')

        self.check_similarity = check_similarity
        self.df_descriptors = pd.read_csv(path_descriptors, index_col='inchi')

        params_files_path = os.listdir(folder_to_read)
        params_dict = dict()
        i = 0
        for file_path in params_files_path:
            path_to_read = os.path.join(folder_to_read, file_path)
            params_dict[f'params_{i}'] = dict(np.load(path_to_read))
            params_dict[f'params_{i}']['seed'] = int(params_dict[f'params_{i}']['seed'])
            params_dict[f'params_{i}']['n_hidden_layers'] = int(params_dict[f'params_{i}']['n_hidden_layers'])
            i += 1
        self.params_dict = params_dict
        self.descriptors_list = descriptors_list

    def get_gammaT(self, inchi0, inchi1, temperature):

        input_i = self.df_descriptors.loc[inchi0, self.descriptors_list]
        assert(len(input_i) == len(self.descriptors_list))
        input_i = input_i.to_dict()
        
        input_j = self.df_descriptors.loc[inchi1, self.descriptors_list]
        assert(len(input_j) == len(self.descriptors_list))
        input_j = input_j.to_dict()

        # scaling the inputs
        out = descriptors_scaling(input_i, input_j, temperature, self.descriptors_list)
        scaled_input_i, scaled_input_j, scaled_temperature = out

        gammas = [self.model(scaled_input_i, scaled_input_j, scaled_temperature, np_params, self.check_similarity)
                  for np_params in self.params_dict.values()]
        return gammas

# The location of the folder containing this file
here = os.path.abspath(os.path.dirname(__file__))

class GammaTModelsDatabase:
    # This is the list of descriptors used to train the model
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']
 
    # A map from model key to its type   
    model_types = {
        'TempDep': 'Tdep', 
        'TempDep_reg4_d2y': 'Tdep', 
        'TempDep_cossim': 'Tdep', 
        'TempDep_cossim_reg4_d2y': 'Tdep', 
        'TempInd': 'Tind', 
        'TempInd_reg4': 'Tind',
        'TempInd_cossim': 'Tind', 
        'TempInd_cossim_reg4': 'Tind'
    }

    def __init__(self, *, 
        model_keys=model_types.keys(), 
        params_folder=(here+'/../params' if os.path.exists(here+'/../params') else here+'/params'),
        path_descriptors=here+'/db_REFPROP_descriptors.csv'
        ):
        """Constructor of the model collection
        
        Sensible defaults are used to load default parameters but can be over-written if needed

        Args:
            model_keys (str, optional): list of model keys to be loaded. Defaults to model_types.keys().
            params_folder (str, optional): Folder containing the parameters. Defaults to (here+'/../params' if os.path.exists(here+'/../params') else here+'/params').
            path_descriptors (str, optional): Descriptors used in the models. Defaults to here+'/db_REFPROP_descriptors.csv'.
        """

        self.models = {}
        for model_key in model_keys:
            folder_to_read = os.path.join(params_folder, f'ANN_models_{model_key}', 'gammaT-REFPROP-LM-average')
            GammaCalculatorObj = GammaCalculator(
                folder_to_read=folder_to_read, 
                path_descriptors=path_descriptors, 
                descriptors_list=self.descriptors_list,
                model_type=self.model_types[model_key],
                check_similarity='cossim' in model_key)
            self.models[model_key] = GammaCalculatorObj

    def add_descriptors_ctREFPROP(self, InChI:str, RP:ct.REFPROPFunctionLibrary):
        """Add descriptors for a fluid to the descriptors database with the ctREFPROP interface for NIST REFPROP

        Args:
            InChI (str): The InChI string for the fluid
            RP (ct.REFPROPFunctionLibrary): The instance of ct.REFPROPFunctionLibrary loaded with the fluid of interest
        """
        
        r = RP.REFPROPdll('', '', 'TC;PC;DC;M;TTRP;ACF;DIPOLE;TNBP;PTRP',RP.MOLAR_BASE_SI,0,0,0,0,[1.0])
        Tc, pc, Dc, MM, TTRP, ACF, DIPOLE, TNBP, PTRP = r.Output[0:9]
        descriptors = {
            'InChI': InChI,
            'TC [K]': Tc,
            "PC [Pa]": pc,
            "DC [mol/m^3]": Dc,
            "MM [kg/mol]": MM,
            "TTRP [K]": TTRP,
            "ACF [-]": ACF,
            "DIPOLE [debye]": DIPOLE,
            'TNBP [K]': TNBP,
            'PTRP [Pa]': PTRP
        }
        # print(descriptors)
        
        for model_key in self.model_types.keys():
            # Take the old DataFrame and merge with a one-element DataFrame (this is quite inefficient, but ok for purposes here)
            orig = self.models[model_key].df_descriptors.copy()
            # Make the new DataFrame with the descriptors to be added
            f2 = pd.DataFrame([descriptors])
            f2.set_index('InChI', inplace=True)
            
            if f2.index[0] in orig.index:
                raise KeyError('Adding an InChI already in the descriptor set: ' + f2.index[0])
            
            # Concatenate them together
            newdf = pd.concat([orig, f2], sort=True)

            # And store the merged db
            self.models[model_key].df_descriptors = newdf

    def add_descriptors_CoolProp(self, InChI, AS):
        pass 
    
    def get_gammaTs(self, *, key:str, InChI_i:str, InChI_j:str, T_K:float):
        """Extract the gammaT values from each of the model trainings

        Args:
            key (str): The model type, one of the keys in model_types
            InChI_i (str): The first InChI string
            InChI_j (str): The second InChI string
            T_K (float): The temperature, always required, sometimes not used depending on the model

        Returns:
            numpy.NDArray: The sequence of obtained values for gammaT
        """
        return self.models[key].get_gammaT(InChI_i, InChI_j, T_K)

if __name__ == '__main__':
    
    db = GammaTModelsDatabase()
    
    if os.getenv('RRPPREFIX') is None:
        os.environ['RPPREFIX'] = os.getenv('HOME') + '/REFPROP10'
    
    root = os.getenv('RPPREFIX')
    RP = ct.REFPROPFunctionLibrary(root)
    RP.SETPATHdll(root)
    INCHIS = []
    for fluid in ['TOLUENE']:
        ierr = RP.SETFLUIDSdll(fluid)
        if ierr != 0: raise ValueError(RP.ERRMSGdll(ierr))
        InChI = 'InChI='+RP.REFPROPdll('', '','INCHI',0,0,0,0,0,[1.0]).hUnits
        db.add_descriptors_ctREFPROP(InChI=InChI, RP=RP)
        INCHIS.append(InChI)
    db.models['TempInd'].df_descriptors.to_csv('ddd.csv')
    
    ierr = RP.SETFLUIDSdll('R1336MZZZ')
    if ierr != 0: raise ValueError(RP.ERRMSGdll(ierr))
    InChI_mzzZ = 'InChI='+RP.REFPROPdll('', '','INCHI',0,0,0,0,0,[1.0]).hUnits

    print(np.mean(db.get_gammaTs(key='TempInd', InChI_i=INCHIS[0], InChI_j=InChI_mzzZ, T_K=300)))
    quit()
    
    # running some examples
    folder_to_read = './'
    filename_descriptors = 'db_REFPROP_descriptors.csv'
    path_descriptors = os.path.join(folder_to_read, filename_descriptors)
    df_descriptors = pd.read_csv(path_descriptors, index_col=0)
    # index to select two inputs
    i = 0
    j = 1
    # Getting inchi of descriptors. They have the same order as descriptor list
    inchi_i = df_descriptors.loc[i, 'inchi']
    inchi_j = df_descriptors.loc[j, 'inchi']
    input_T = 350. # K 

    preferred_name_i = df_descriptors.loc[i, 'preferred_name']
    preferred_name_j = df_descriptors.loc[j, 'preferred_name']
    print(f"Name of input i: {preferred_name_i}")
    print(f"Name of input j: {preferred_name_j}")

    
    params_folder = '../params'
    for model_key in ['TempDep', 'TempDep_reg4_d2y', 'TempDep_cossim', 'TempDep_cossim_reg4_d2y', 
                      'TempInd', 'TempInd_reg4', 'TempInd_cossim', 'TempInd_cossim_reg4']:
        ANN_params_folder = f'ANN_models_{model_key}'
        prefix = 'gammaT-REFPROP-LM-average'
        folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
        # This is the list of descriptors
        descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

        if 'cossim' in model_key:
            check_similarity = True
        else:
            check_similarity = False

        GammaCalculatorObj = GammaCalculator(
            folder_to_read=folder_to_read, 
            path_descriptors=path_descriptors, 
            descriptors_list=descriptors_list,
            model_type=GammaTModelsDatabase.model_types[model_key],
            check_similarity=check_similarity)
        gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_i, input_T)
        gammaT_mean = np.mean(gammas)
        gammaT_std = np.std(gammas)
        print("-------------------")
        print("ANN type:", ANN_params_folder)
        print("Model:", prefix)
        print("Mean value of gammaT: ", gammaT_mean)
        print("Standard deviation of gammaT: ", gammaT_std)
        print("-------------------")
    """
    ###########
    # Example 1
    # reading parameters
    model_type = 'reg0'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read, 
                                        path_descriptors=path_descriptors, 
                                        descriptors_list=descriptors_list,
                                        model_type='Tdep')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")

    ###########
    # Example 2
    # reading parameters
    model_type = 'reg3'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read, 
                                        path_descriptors=path_descriptors, 
                                        descriptors_list=descriptors_list,
                                        model_type='Tdep')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")


    ###########
    # Example 3
    # reading parameters
    model_type = 'linear'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read, 
                                        path_descriptors=path_descriptors, 
                                        descriptors_list=descriptors_list,
                                        model_type='linear')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")

    ###########
    # Example 4
    # reading parameters
    model_type = 'd2y'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read,
                                        path_descriptors=path_descriptors,
                                        descriptors_list=descriptors_list,
                                        model_type='Tdep')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")

    ###########
    # Example 5
    # reading parameters
    model_type = 'TempInd'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read,
                                        path_descriptors=path_descriptors,
                                        descriptors_list=descriptors_list,
                                        model_type='Tind')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")

    ###########
    # Example 6
    # reading parameters
    model_type = 'dy_d2y'
    ANN_params_folder = f'ANN_models_{model_type}'
    prefix = 'gammaT-REFPROP-LM-average'
    folder_to_read = os.path.join(params_folder, ANN_params_folder, prefix)
    # This is the list of descriptors
    descriptors_list = ['TC [K]', 'PC [Pa]', 'ACF [-]', 'DIPOLE [debye]']

    GammaCalculatorObj = GammaCalculator(folder_to_read=folder_to_read,
                                         path_descriptors=path_descriptors,
                                         descriptors_list=descriptors_list,
                                         model_type='Tdep')
    gammas = GammaCalculatorObj.get_gammaT(inchi_i, inchi_j, input_T)
    gammaT_mean = np.mean(gammas)
    gammaT_std = np.std(gammas)
    print("-------------------")
    print("ANN type:", ANN_params_folder)
    print("Model:", prefix)
    print("Mean value of gammaT: ", gammaT_mean)
    print("Standard deviation of gammaT: ", gammaT_std)
    print("-------------------")
    """