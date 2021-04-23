import logging
import re
import pysmiles
from typing import List


# Apprentissage avec le fichier train.txt
def train(model_name: str, gamma: float, model_number: str, regularization_coef: float, batch_size: int, hidden_dim: int, lr: float, neg_sample_size: int, num_thread: int = 1, num_proc: int = 1, max_step: int = 10000):
    os.system("DGLBACKEND=pytorch dglke_train"
              ' --dataset BASE'
              f' --model_name {model_name}'
              ' --data_path ./train'
              ' --data_files train.txt valid.txt test.txt'
              ' --format \'raw_udd_hrt\''
              f' --batch_size {batch_size}'
              f' --neg_sample_size {neg_sample_size}'
              f' --hidden_dim {hidden_dim}'
              f' --gamma {gamma}'
              f' --lr {lr}'
              f' --max_step {max_step}'
              ' --log_interval 100'
              ' --batch_size_eval 16'
              ' -adv'
              f' --regularization_coef {regularization_coef}'
              f' --num_thread {num_thread}'
              f' --num_proc {num_proc}'
              f' --save_path ./ckpts/{model_number}')


# Prediction avec le fichier test.txt
def predict(model_number: str, score_func: str = 'logsigmoid'):
    os.system(f' dglke_predict --model_path ckpts/{model_number}/TransE_l2_BASE_0/'
              " --format 'h_r_t' "
              " --data_files predict/head.list predict/rel.list predict/tail.list"
              ' --raw_data'
              ' --entity_mfile train/entities.tsv'
              ' --rel_mfile train/relations.tsv'
              f' --score_func {score_func}'
              " --exec_mode 'all'"
              ' --topK 10'
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


def get_id(description: str) -> str:
    split = description.split('\n', -1)
    identifier = next(el for el in split if el != '')
    return identifier


def get_smile(description: str) -> str:
    search = re.findall(r'> {2}<Smiles>\n[\s\S]*$\n', description)[0]
    split = search.split('\n', -1)
    return split[1]


def molecule_to_train(molecule: dict, column_separator: str) -> str:
    # A partir du smiles, on construit le graphe de connaissance
    graph = pysmiles.read_smiles(molecule['smiles'])
    atoms = graph.nodes(data='element')
    
    content: str = ''
    for edge in graph.edges:
        edge_from = f'{molecule["name"]}_{edge[0]}'
        edge_to = f'{molecule["name"]}_{edge[1]}'

        content += f'{edge_from}{column_separator}connected_to{column_separator}{edge_to}\n'

        content += f'{edge_from}{column_separator}chemical_element{column_separator}{atoms[edge[0]]}\n'
        content += f'{edge_to}{column_separator}chemical_element{column_separator}{atoms[edge[1]]}\n'

    for index, atom in enumerate(atoms):
        content += f'{molecule["name"]}{column_separator}contains{column_separator}{molecule["name"]}_{index}\n'

    return content


def set_to_train(source_set: List[dict], column_separator: str = '\t') -> str:
    """
    Transforme une liste de molécules en un graphe de connaissance
    Parameters
    ----------
    source_set: list[dict]
        liste de molécules sous forme {'name': 'foo', 'smiles': 'bar}

    Returns
    -------
    str
        le contenu du graphe de connaissance, sous forme de string
    """
    content: str = ''
    for molecule in source_set:
        content += molecule_to_train(molecule=molecule, column_separator=column_separator)
        
    return content


def write(content: str, file_path: str):
    with open(file_path, 'w+') as file:
        file.write(content)


def set_to_valid(source_set: List[dict], column_separator: str = '\t') -> str:
    content: str = ''
    for molecule in source_set:
        content += f'{molecule["name"]}{column_separator}cure{column_separator}Covid-19\n'

    return content


def set_to_head(source_set: List[dict]) -> str:
    content: str = ''
    for molecule in source_set:
        content += f'{molecule["name"]}\n'

    return content


def remove_duplicate_molecule(source_set: List[dict]) -> List[dict]:
    destination_set: List[dict] = []
    for molecule in source_set:
        if molecule not in destination_set:
            destination_set.append(molecule)

    return destination_set


def get_scores(file_path: str, column_separator: str = '\t') -> List[dict]:
    content: list[dict] = []
    with open(file=file_path, mode='r') as file:
        file.readline()
        while True:
            line: str = file.readline()

            if not line:
                break

            columns: [str] = line.split(sep=column_separator)
            content.append({ 'name': columns[0], 'score': float(columns[3]) })

    return content


def get_best_scores(scores: List[dict], count: int = 10):
    max_el = max(scores, key=lambda x: x['score'])
    print(max_el)


if __name__ == '__main__':
    # Main function
    logging.getLogger('pysmiles').setLevel(logging.CRITICAL)

    
    # Extraire les données des fichiers SDF
    training_set: List[dict] = extract(file_path='data/main_protease_inhibitors.sdf')
    test_set: List[dict] = extract(file_path='data/LC_Protease.sdf')

    # Transformer un set de molécule en un fichier de train
    train_dot_txt = set_to_train(source_set=training_set+test_set)

    # Ecrire le graph dans un fichier de train
    write(content=train_dot_txt, file_path='train/train.txt')
    
    # On donne les relations valides
    valid_dot_txt = set_to_valid(source_set=training_set)
    write(content=valid_dot_txt, file_path='train/valid.txt')
    write(content=valid_dot_txt, file_path='train/test.txt')

    # Phase d'apprentissage avec le fichier d'apprentissage créer
    train(model_name="TransE_l2", gamma=12.0, model_number="model1", regularization_coef=1.00E-09, batch_size=20, hidden_dim=40, lr=0.25, neg_sample_size=2, num_thread=6, num_proc=1, max_step=5000)

    # On donne la liste des molécules à tester
    head_dot_list = set_to_head(source_set=test_set)
    write(content=head_dot_list, file_path='predict/head.list')
    
    write(content="cure\n", file_path='predict/rel.list')
    write(content="Covid-19\n", file_path='predict/tail.list')

    # Phase de prédiction 
    predict("model1")

    # Trier les résultats et récuperer les meilleurs
    # scores = get_scores(file_path='ckpts/model1/result.tsv')
    # get_best_scores(scores=scores)
