import sys
sys.path.append('/home/aistudio/work/paddle_project/utils')
import paddle_aux
import os
import paddle
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree


def split_data(dataset, train_ratio, val_ratio, test_ratio, seed=np.random.
    randint(1, 1000000.0), save=False):
    dataset_size = len(dataset)
    if train_ratio + val_ratio + test_ratio <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        train_dataset, val_dataset, test_dataset, unused_dataset = (paddle.
            io.random_split(dataset=dataset, lengths=[train_length,
            val_length, test_length, unused_length]))
        print('train length:', train_length, 'val length:', val_length,
            'test length:', test_length, 'unused length:', unused_length,
            'seed :', seed)
        return train_dataset, val_dataset, test_dataset
    else:
        print('invalid ratios')


def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1000000.0
    ), save=False):
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for i in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = paddle.io.random_split(dataset=dataset, lengths=folds)
    print('fold length :', fold_length, 'unused length:', unused_length,
        'seed', seed)
    return cv_dataset[0:num_folds]


def get_dataset(data_path, target_index, reprocess='False', processing_args
    =None):
    if processing_args == None:
        processed_path = 'processed'
    else:
        processed_path = processing_args.get('processed_path', 'processed')
    transforms = GetY(index=target_index)
    if os.path.exists(data_path) == False:
        print('Data not found in:', data_path)
        sys.exit()
    if reprocess == 'True':
        os.system('rm -rf ' + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)
    if os.path.exists(os.path.join(data_path, processed_path, 'data.pt')
        ) == True:
        dataset = StructureDataset(data_path, processed_path, transforms)
    elif os.path.exists(os.path.join(data_path, processed_path, 'data0.pt')
        ) == True:
        dataset = StructureDataset_large(data_path, processed_path, transforms)
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, 'data.pt')
            ) == True:
            dataset = StructureDataset(data_path, processed_path, transforms)
        elif os.path.exists(os.path.join(data_path, processed_path, 'data0.pt')
            ) == True:
            dataset = StructureDataset_large(data_path, processed_path,
                transforms)
    return dataset


class StructureDataset(InMemoryDataset):

    def __init__(self, data_path, processed_path='processed', transform=
        None, pre_transform=None):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform,
            pre_transform)
        self.data, self.slices = paddle.load(path=str(self.processed_paths[0]))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ['data.pt']
        return file_names


class StructureDataset_large(Dataset):

    def __init__(self, data_path, processed_path='processed', transform=
        None, pre_transform=None):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(data_path, transform,
            pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = []
        for file_name in glob.glob(self.processed_dir + '/data*.pt'):
            file_names.append(os.path.basename(file_name))
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = paddle.load(path=str(os.path.join(self.processed_dir,
            'data_{}.pt'.format(idx))))
        return data


def create_global_feat(atoms_index_arr):
    comp = np.zeros(108)
    temp = np.unique(atoms_index_arr, return_counts=True)
    for i in range(len(temp[0])):
        comp[temp[0][i]] = temp[1][i] / temp[1].sum()
    return comp.reshape(1, -1)


def process_data(data_path, processed_path, processing_args):
    print('Processing data to: ' + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), 'Data path not found in ' + data_path
    if processing_args['dictionary_source'] != 'generated':
        if processing_args['dictionary_source'] == 'default':
            print('Using default dictionary.')
            atom_dictionary = get_dictionary(os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'dictionary_default.json'))
        elif processing_args['dictionary_source'] == 'blank':
            print(
                'Using blank dictionary. Warning: only do this if you know what you are doing'
                )
            atom_dictionary = get_dictionary(os.path.join(os.path.dirname(
                os.path.realpath(__file__)), 'dictionary_blank.json'))
        else:
            dictionary_file_path = os.path.join(data_path, processing_args[
                'dictionary_path'])
            if os.path.exists(dictionary_file_path) == False:
                print('Atom dictionary not found, exiting program...')
                sys.exit()
            else:
                print('Loading atom dictionary from file.')
                atom_dictionary = get_dictionary(dictionary_file_path)
    target_property_file = os.path.join(data_path, processing_args[
        'target_path'])
    assert os.path.exists(target_property_file
        ), 'targets not found in ' + target_property_file
    with open(target_property_file) as f:
        reader = csv.reader(f)
        target_data = [row for row in reader]
    ase_crystal_list = []
    if processing_args['data_format'] == 'db':
        db = ase.db.connect(os.path.join(data_path, 'data.db'))
        row_count = 0
        for row in db.select():
            ase_temp = row.toatoms()
            ase_crystal_list.append(ase_temp)
            row_count = row_count + 1
            if row_count % 500 == 0:
                print('db processed: ', row_count)
    data_list = []
    for index in range(0, len(target_data)):
        structure_id = target_data[index][0]
        data = Data()
        if processing_args['data_format'] != 'db':
            ase_crystal = ase.io.read(os.path.join(data_path, structure_id +
                '.' + processing_args['data_format']))
            data.ase = ase_crystal
        else:
            ase_crystal = ase_crystal_list[index]
            data.ase = ase_crystal
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))
        distance_matrix = ase_crystal.get_all_distances(mic=True)
        distance_matrix_trimmed = threshold_sort(distance_matrix,
            processing_args['graph_max_radius'], processing_args[
            'graph_max_neighbors'], adj=False)
        distance_matrix_trimmed = paddle.to_tensor(data=distance_matrix_trimmed
            )
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(edge_index,
                edge_weight, num_nodes=len(ase_crystal), fill_value=0)
            data.edge_index = edge_index
            data.edge_weight = edge_weight
            distance_matrix_mask = (distance_matrix_trimmed.fill_diagonal_(
                value=1) != 0).astype(dtype='int32')
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight
            distance_matrix_mask = (distance_matrix_trimmed != 0).astype(dtype
                ='int32')
        data.edge_descriptor = {}
        data.edge_descriptor['distance'] = edge_weight
        data.edge_descriptor['mask'] = distance_matrix_mask
        target = target_data[index][1:]
        y = paddle.to_tensor(data=np.array([target], dtype=np.float32),
            dtype='float32')
        data.y = y
        _atoms_index = ase_crystal.get_atomic_numbers()
        gatgnn_glob_feat = create_global_feat(_atoms_index)
        gatgnn_glob_feat = np.repeat(gatgnn_glob_feat, len(_atoms_index),
            axis=0)
        data.glob_feat = paddle.to_tensor(data=gatgnn_glob_feat).astype(dtype
            ='float32')
        z = paddle.to_tensor(data=ase_crystal.get_atomic_numbers(), dtype=
            'int64')
        data.z = z
        u = np.zeros(3)
        u = paddle.to_tensor(data=u[np.newaxis, ...], dtype='float32')
        data.u = u
        data.structure_id = [[structure_id] * len(data.y)]
        if processing_args['verbose'] == 'True' and ((index + 1) % 500 == 0 or
            index + 1 == len(target_data)):
            print('Data processed: ', index + 1, 'out of', len(target_data))
        data_list.append(data)
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    paddle.sort(x=species), paddle.argsort(x=species)
    num_species = len(species)
    if processing_args['verbose'] == 'True':
        print('Max structure size: ', n_atoms_max,
            'Max number of elements: ', num_species)
        print('Unique species:', species)
    crystal_length = len(ase_crystal)
    data.length = paddle.to_tensor(data=[crystal_length], dtype='int64')
    if processing_args['dictionary_source'] != 'generated':
        for index in range(0, len(data_list)):
            atom_fea = np.vstack([atom_dictionary[str(data_list[index].ase.
                get_atomic_numbers()[i])] for i in range(len(data_list[
                index].ase))]).astype(float)
            data_list[index].x = paddle.to_tensor(data=atom_fea)
    elif processing_args['dictionary_source'] == 'generated':
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        lb.fit(species)
        for index in range(0, len(data_list)):
            data_list[index].x = paddle.to_tensor(data=lb.transform(
                data_list[index].ase.get_chemical_symbols()), dtype='float32')
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(data_list[index], processing_args[
            'graph_max_neighbors'] + 1)
    processing_args['voronoi'] = 'False'
    if processing_args['voronoi'] == 'True':
        from pymatgen.core.structure import Structure
        from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
        from pymatgen.io.ase import AseAtomsAdaptor
        Converter = AseAtomsAdaptor()
        for index in range(0, len(data_list)):
            pymatgen_crystal = Converter.get_structure(data_list[index].ase)
            Voronoi = VoronoiConnectivity(pymatgen_crystal, cutoff=
                processing_args['graph_max_radius'])
            connections = Voronoi.max_connectivity
            distance_matrix_voronoi = threshold_sort(connections, 9999,
                processing_args['graph_max_neighbors'], reverse=True, adj=False
                )
            distance_matrix_voronoi = paddle.to_tensor(data=
                distance_matrix_voronoi)
            out = dense_to_sparse(distance_matrix_voronoi)
            edge_index_voronoi = out[0]
            edge_weight_voronoi = out[1]
            edge_attr_voronoi = distance_gaussian(edge_weight_voronoi)
            edge_attr_voronoi = edge_attr_voronoi.astype(dtype='float32')
            data_list[index].edge_index_voronoi = edge_index_voronoi
            data_list[index].edge_weight_voronoi = edge_weight_voronoi
            data_list[index].edge_attr_voronoi = edge_attr_voronoi
            if index % 500 == 0:
                print('Voronoi data processed: ', index)
    if processing_args['SOAP_descriptor'] == 'True':
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False
        from dscribe.descriptors import SOAP
        make_feature_SOAP = SOAP(species=species, rcut=processing_args[
            'SOAP_rcut'], nmax=processing_args['SOAP_nmax'], lmax=
            processing_args['SOAP_lmax'], sigma=processing_args[
            'SOAP_sigma'], periodic=periodicity, sparse=False, average=
            'inner', rbf='gto', crossover=False)
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = paddle.to_tensor(data=
                features_SOAP)
            if processing_args['verbose'] == 'True' and index % 500 == 0:
                if index == 0:
                    print('SOAP length: ', tuple(features_SOAP.shape))
                print('SOAP descriptor processed: ', index)
    elif processing_args['SM_descriptor'] == 'True':
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False
        from dscribe.descriptors import SineMatrix, CoulombMatrix
        if periodicity == True:
            make_feature_SM = SineMatrix(n_atoms_max=n_atoms_max,
                permutation='eigenspectrum', sparse=False, flatten=True)
        else:
            make_feature_SM = CoulombMatrix(n_atoms_max=n_atoms_max,
                permutation='eigenspectrum', sparse=False, flatten=True)
        for index in range(0, len(data_list)):
            features_SM = make_feature_SM.create(data_list[index].ase)
            data_list[index].extra_features_SM = paddle.to_tensor(data=
                features_SM)
            if processing_args['verbose'] == 'True' and index % 500 == 0:
                if index == 0:
                    print('SM length: ', tuple(features_SM.shape))
                print('SM descriptor processed: ', index)
    if processing_args['edge_features'] == 'True':
        distance_gaussian = GaussianSmearing(0, 1, processing_args[
            'graph_edge_length'], 0.2)
        NormalizeEdge(data_list, 'distance')
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(data_list[index]
                .edge_descriptor['distance'])
            if processing_args['verbose'] == 'True' and ((index + 1) % 500 ==
                0 or index + 1 == len(target_data)):
                print('Edge processed: ', index + 1, 'out of', len(target_data)
                    )
    Cleanup(data_list, ['ase', 'edge_descriptor'])
    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))
    if processing_args['dataset_type'] == 'inmemory':
        data, slices = InMemoryDataset.collate(data_list)
        paddle.save(obj=(data, slices), path=os.path.join(data_path,
            processed_path, 'data.pt'))
    elif processing_args['dataset_type'] == 'large':
        for i in range(0, len(data_list)):
            paddle.save(obj=data_list[i], path=os.path.join(os.path.join(
                data_path, processed_path), 'data_{}.pt'.format(i)))


def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed, method=
            'ordinal', axis=1)
    elif reverse == True:
        distance_matrix_trimmed = rankdata(distance_matrix_trimmed * -1,
            method='ordinal', axis=1)
    distance_matrix_trimmed = np.nan_to_num(np.where(mask, np.nan,
        distance_matrix_trimmed))
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0
    if adj == False:
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0,
            distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((tuple(matrix.shape)[0], neighbors + 1))
        adj_attr = np.zeros((tuple(matrix.shape)[0], neighbors + 1))
        for i in range(0, tuple(matrix.shape)[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(temp, pad_width=(0, neighbors + 1 - len
                (temp)), mode='constant', constant_values=0)
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(distance_matrix_trimmed == 0,
            distance_matrix_trimmed, matrix)
        return distance_matrix_trimmed, adj_list, adj_attr


class GaussianSmearing(paddle.nn.Layer):

    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs
        ):
        super(GaussianSmearing, self).__init__()
        offset = paddle.linspace(start=start, stop=stop, num=resolution)
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer(name='offset', tensor=offset)

    def forward(self, dist):
        dist = dist.unsqueeze(axis=-1) - self.offset.view(1, -1)
        return paddle.exp(x=self.coeff * paddle.pow(x=dist, y=2))


def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype='int64')
    deg = paddle.nn.functional.one_hot(num_classes=max_degree + 1, x=deg
        ).astype('int64').to('float32')
    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = paddle.concat(x=[x, deg.to(x.dtype)], axis=-1)
    else:
        data.x = deg
    return data


def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label
                    ].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label
                    ].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max(
                ) > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label
                    ].max()
            if dataset[index].edge_descriptor[descriptor_label].min(
                ) < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label
                    ].min()
    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)
    for data in dataset:
        data.edge_descriptor[descriptor_label] = (data.edge_descriptor[
            descriptor_label] - feature_min) / (feature_max - feature_min)


def SM_Edge(dataset):
    from dscribe.descriptors import CoulombMatrix, SOAP, MBTR, EwaldSumMatrix, SineMatrix
    count = 0
    for data in dataset:
        n_atoms_max = len(data.ase)
        make_feature_SM = SineMatrix(n_atoms_max=n_atoms_max, permutation=
            'none', sparse=False, flatten=False)
        features_SM = make_feature_SM.create(data.ase)
        features_SM_trimmed = np.where(data.mask == 0, data.mask, features_SM)
        features_SM_trimmed = paddle.to_tensor(data=features_SM_trimmed)
        out = dense_to_sparse(features_SM_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        data.edge_descriptor['SM'] = edge_weight
        if count % 500 == 0:
            print('SM data processed: ', count)
        count = count + 1
    return dataset


class GetY(object):

    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data
