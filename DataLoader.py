import os

import numpy as np

from sklearn.datasets import make_circles, make_moons, make_blobs, make_swiss_roll, make_s_curve

class DataLoader:
    def __init__(self) -> None:
        self.dataset_types = ['circles', 'moons', 'blobs', 'aniso', 'varied', 'swiss_roll', 's_curve']

    def _load_data_from_sklearn(self, name: str, 
                               num_samples: int = 50, random_state:int = 0, 
                               **kwargs) -> tuple:
        match name:
            case 'circles':
                # Extract specific parameters for 'circles', with defaults
                noise_std = kwargs.get('noise_std', 0.5)
                factor = kwargs.get('factor', 0.5)
                circles = make_circles(n_samples=num_samples,
                                    noise=noise_std,
                                    factor=factor,
                                    random_state=random_state)
                return (circles,
                            {
                                'n_samples': num_samples,
                                'noise': noise_std,
                                'factor': factor,
                                'random_state': random_state
                            }
                        )
            
            case 'moons':
                # Extract specific parameters for 'moons', with defaults
                noise_std = kwargs.get('noise_std', 0.5)
                moons = make_moons(n_samples=num_samples,
                                    noise=noise_std,
                                    random_state=random_state)
                return (moons,
                            {
                                'n_samples': num_samples,
                                'noise': noise_std,
                                'random_state': random_state
                            }
                        )
            
            case 'blobs':
                # No specific parameters for 'blobs', just defaults
                blobs = make_blobs(n_samples=num_samples,
                                    random_state=random_state)
                return (blobs,
                            {
                                'n_samples': num_samples,
                                'random_state': random_state
                            }
                        )
            
            case 'aniso':
                # Extract specific parameters for 'aniso' (Anisotropicly distributed data), with defaults
                # random_state = 170 # maybe do this?
                noise_std = kwargs.get('noise_std', 0.5)
                X, y = make_blobs(n_samples=num_samples,
                                    cluster_std=noise_std,
                                    random_state=random_state)
                
                transformation = [[0.6, -0.6], [-0.4, 0.8]]
                X_aniso = np.dot(X, transformation)
                aniso = (X_aniso, y)
                return (aniso,
                            {
                                'n_samples': num_samples,
                                'cluster_std': noise_std,
                                'random_state': random_state
                            }
                        )
                
            case 'varied':
                # Extract specific parameters for 'varied' (Blobs with varied variances), with defaults
                noise_std = kwargs.get('noise_std', 0.5)
                varied = make_blobs(n_samples=num_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
                return (varied,
                            {
                                'n_samples': num_samples,
                                'cluster_std': noise_std,
                                'random_state': random_state
                            }
                        )
            
            case 'swiss_roll':
                # Extract specific parameters for 'swiss_roll', with defaults
                noise_std = kwargs.get('noise_std', 0.5)
                swiss_roll = make_swiss_roll(n_samples=num_samples,
                                    noise=noise_std,
                                    random_state=random_state)
                return (swiss_roll,
                            {
                                'n_samples': num_samples,
                                'noise': noise_std,
                                'random_state': random_state
                            }
                        )
            
            case 's_curve':
                # Extract specific parameters for 's_curve', with defaults
                noise_std = kwargs.get('noise_std', 0.5)
                s_curve = make_s_curve(n_samples=num_samples,
                                    noise=noise_std,
                                    random_state=random_state)
                return (s_curve,
                            {
                                'n_samples': num_samples,
                                'noise': noise_std,
                                'random_state': random_state
                            }
                        )
            case _:
                raise Exception(f"Dataset {name} not found / not implemented.")

    def load_data(self,  names: list = ['all'], 
                    num_samples: int = 50, random_state: int = 0,
                    **kwargs):
        ''' 
        Loads data from sklearn.datasets.
        '''
        names = self.dataset_types if names[0] == 'all' else names

        # if not os.path.exists('Data'):
        #     os.makedirs('Data')

        datasets = []
        for name in names:
            
            # if not os.path.exists(f"Data/{name}"):
            #     os.makedirs(f"Data/{name}")
                
            dataset = self._load_data_from_sklearn(name, **kwargs)

            datasets.append(dataset)

        return datasets

        

    