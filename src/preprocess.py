import pandas as pd
import numpy as np
import os 

path = os.path.join('/home/monerl/Projekty/Neuroguard', 'data/')

columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
    ]


def preprocess(file_path, save, columns):
    
    df = pd.read_csv(path + file_path, names = columns, header = None)

    df['label'].where(df['label'] == 'normal', inplace = True)
    df['label'].fillna(value = 1, inplace = True)
    df.loc[df['label'] == 'normal', 'label' ] = 0

    df = pd.get_dummies(df, columns = ['protocol_type', 'service', 'flag'])

    df.astype(np.float32)

    x = df.drop(columns = ['label', 'difficulty_level'])
    y = df['label']

    x_array = x.astype(np.float32).values
    y_array = y.astype(np.float32).values

    processed_dir = path + save

    np.save(f'{processed_dir}/X_train.npy', x_array)
    np.save(f'{processed_dir}/y_train.npy', y_array)

preprocess('KDDTrain+.txt', 'processed', columns)
