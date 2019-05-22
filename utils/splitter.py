
import numpy as np
import json

def val_test_split(validation_filename, sub_val_filename, sub_test_filename, test_fraction = 0.25):
    json_start = json.load(open(validation_filename, 'r'))
    
    dict_validation = dict()
    dict_test = dict()
    
    for key in json_start.keys():
        samples = np.array(json_start[key])
        indices_test = np.random.choice(samples.size, size = int(np.floor(test_fraction * samples.size)))
        indices_test_bool = np.array([i in indices_test for i in range(len(samples))])
        dict_validation[key] = samples[~indices_test_bool].tolist()
        dict_test[key] = samples[indices_test_bool].tolist()
    
    json.dump(dict_validation, open(sub_val_filename, 'w'))
    json.dump(dict_test, open(sub_test_filename, 'w'))
        
    