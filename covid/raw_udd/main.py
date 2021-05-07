import logging
import re
import pysmiles
import os
import numpy
from typing import List
from pprint import pprint


# Apprentissage avec le fichier train.txt
def train(model_name: str, model_number: str, regularization_coef: float, batch_size: int, hidden_dim: int, neg_sample_size: int, num_thread: int = 1, num_proc: int = 1, max_step: int = 10000):
    os.system("DGLBACKEND=pytorch dglke_train"
              ' --dataset BASE'
              f' --model_name {model_name}'
              ' --data_path ./train'
              ' --data_files train.tsv valid.tsv test.tsv'
              ' --format \'raw_udd_hrt\''
              f' --batch_size {batch_size}'
              f' --neg_sample_size {neg_sample_size}'
              f' --hidden_dim {hidden_dim}'
              f' --max_step {max_step}'
              ' --log_interval 100'
              ' --batch_size_eval 16'
			  f' --regularization_coef {regularization_coef}'
              f' --num_thread {num_thread}'
              f' --num_proc {num_proc}'
              f' --save_path ./ckpts/{model_number}')


# Prediction avec le fichier test.txt
def predict(model_name: str, model_number: str, score_func: str = 'logsigmoid'):
    os.system(f' dglke_predict --model_path ckpts/{model_number}/{model_name}_BASE_0/'
              " --format 'h_r_t' "
              " --data_files predict/head.list predict/rel.list predict/tail.list"
              ' --raw_data'
              ' --entity_mfile train/entities.tsv'
              ' --rel_mfile train/relations.tsv'
              f' --score_func {score_func}'
              " --exec_mode 'all'"
              ' --topK 20'
              f' --output ckpts/{model_number}/result.tsv')


def extract(file_path: str) -> List[dict]:
    """
    Extract molecules_list description from `.sdf` file

    Parameters
    ----------
    file_path: str
        path to a `.sdf` file

    Returns
    -------
    list
        A list of each molecule description
    """
    molecules: List = []

    with open(file_path, 'r') as file:
        descriptions = split_descriptions(file.read())
        descriptions = filter(lambda x: x != '' and x != '\n', descriptions)

        for description in descriptions:
            # On récupère le nom de la molécule et son smiles
            molecules.append({
                'name': get_id(description),
                'smiles': get_smile(description),
            })

    print(f'extracted {str(len(molecules))} molecules from {file_path}')
    return molecules


def split_descriptions(content: str, separator: str = '$$$$') -> List[str]:
    return content.split(separator, -1)


def get_smile(description: str) -> str:
    search = re.findall(r'> {2}<Smiles>\n[\s\S]*$\n', description)[0]
    split = search.split('\n', -1)
    return split[1]


mapping: [str] = []


def get_id(description: str) -> str:
    name = next(s for s in description.split('\n', -1) if s)
    mapping.append(name)
    return f'{len(mapping) - 1}'


def data_to_triplets(data_set: dict, cure: bool = False) -> [[str]]:
    triplets: [[str]] = []
    for molecule in data_set:
        triplet = molecule_to_triplet(molecule=molecule)
        if cure:
            triplet += [[f'{molecule["name"]}', 'cure', 'Covid-19']]
        triplets += triplet

    return triplets


def molecule_to_triplet(molecule: dict) -> [[str]]:
    graph = pysmiles.read_smiles(molecule['smiles'])
    atoms = graph.nodes(data='element')

    triplets: [[str]] = []
    for node in graph.nodes:
        edge_from = f'{molecule["name"]}_{node}'
        element = graph.nodes(data='element')[node]
        triplets.append([edge_from, 'chemial_element', element])

        for to in graph[node]:
            edge_to = f'{molecule["name"]}_{to}'
            order = graph[node].get(to).get('order')

            link = ""
            if order == 2:
                link = "double_"
            if order == 3:
                link = "triple_"

            triplets.append([edge_from, f'{link}connected_to', edge_to])

    for index, atom in enumerate(atoms):
        triplets.append([molecule["name"], 'contains',
                        f'{molecule["name"]}_{index}'])

    return triplets


def write(content: str, file_path: str):
    with open(file_path, 'w+') as file:
        file.write(content)


def set_to_head(source_set: List[dict]) -> str:
    content: str = ''
    for molecule in source_set:
        content += f'{molecule["name"]}\n'

    return content


def get_scores(file_path: str, column_separator: str = '\t') -> [[str]]:
    triplets: [[str]] = []
    with open(file=file_path, mode='r') as file:
        file.readline()
        while True:
            line: str = file.readline()

            if not line:
                break

            columns: [str] = line.split(sep=column_separator)
            triplets.append([
                mapping[
                    int(columns[0])
                ],
                columns[3]
            ])

    return triplets


if __name__ == '__main__':
    # Main function
    logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

    # Extraire les données des fichiers SDF
    main_protease_set: List[dict] = extract(
        file_path='data/main_protease_inhibitors.sdf')
    lc_protease_set: List[dict] = extract(file_path='data/LC_Protease.sdf')
    triplets = data_to_triplets(data_set=main_protease_set, cure=True)

    triplets += data_to_triplets(data_set=lc_protease_set)
    triplets_count = len(triplets)

	# Seed https://www.dgl.ai/news/2020/06/09/covid.html
    seed = numpy.arange(triplets_count)
    numpy.random.shuffle(seed)

    train_count = int(triplets_count * 0.9)
    valid_count = int(triplets_count * 0.05)

    train_set = seed[:train_count].tolist()
    valid_set = seed[train_count:train_count + valid_count].tolist()
    test_set = seed[train_count + valid_count:].tolist()

    separator = '\t'

    with open('train/train.tsv', 'w+') as f:
        for idx in train_set:
            f.writelines(
                f'{triplets[idx][0]}{separator}{triplets[idx][1]}{separator}{triplets[idx][2]}\n')

    with open('train/valid.tsv', 'w+') as f:
        for idx in valid_set:
            f.writelines(
                f'{triplets[idx][0]}{separator}{triplets[idx][1]}{separator}{triplets[idx][2]}\n')

    with open('train/test.tsv', 'w+') as f:
        for idx in test_set:
            f.writelines(
                f'{triplets[idx][0]}{separator}{triplets[idx][1]}{separator}{triplets[idx][2]}\n')

    model = 'TransE'

    # Phase d'apprentissage avec le fichier d'apprentissage créer
    train(model_name=model, model_number="model1", regularization_coef=1.00E-07,
          batch_size=256, hidden_dim=40, neg_sample_size=32, num_thread=6, num_proc=1, max_step=5000)

    # On donne la liste des molécules à tester
    head_dot_list = set_to_head(source_set=lc_protease_set)
    write(content=head_dot_list, file_path='predict/head.list')

    write(content="cure\n", file_path='predict/rel.list')
    write(content="Covid-19\n", file_path='predict/tail.list')

    # Phase de prédiction
    predict(model_name=model, model_number="model1")

    # Trier les résultats et récuperer les meilleurs
    scores = get_scores(file_path='ckpts/model1/result.tsv')
    with open('results.tsv', 'w+') as f:
        for score in scores:
            f.writelines(f'{score[0]}\t{score[1]}')
