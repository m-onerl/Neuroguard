import os 
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(name)s = %(levelname)s - %(message)s'
)
attack_map = {
    'normal': 0,

    # DoS
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1,
    'teardrop': 1, 'apache2': 1, 'udpstorm': 1, 'processtable': 1,
    'worm': 1, 'mailbomb': 1,

    # Probe
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2,
    'mscan': 2, 'saint': 2,

    # R2L
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3,
    'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'sendmail': 3, 'named': 3, 'snmpgetattack': 3, 'snmpguess': 3,
    'xlock': 3, 'xsnoop': 3, 'httptunnel': 3,

    # U2R
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4,
    'sqlattack': 4, 'xterm': 4, 'ps': 4,
}

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
path = os.path.join('data/')

class DataPreprocesor:
    @staticmethod
    def preprocesor(file_path, save):
        logger.info("Preprocesor starting work..")
        df = pd.read_csv(path + file_path, names = columns, header = None)

        df['label'] = df['label'].map(attack_map)

        df = pd.get_dummies(df, columns = ['protocol_type', 'service', 'flag'])

        x = df.drop(columns = ['label', 'difficulty_level']).astype(np.float32)
        y = df['label'].astype(np.float32)

        x_array = x.values
        y_array = y.values

        processed_dir = path + save

        np.save(f'{processed_dir}/X_train.npy', x_array)
        np.save(f'{processed_dir}/y_train.npy', y_array)
        logger.info(f"Files saved into {processed_dir} ")

