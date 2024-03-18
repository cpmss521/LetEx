
import json
from datasets import load_dataset
from datasets import DatasetDict, concatenate_datasets

DATASET_ROOT = 'datasets'



class DatasetLoader(object):
    def __init__(self, dataset_name, has_valid, llm, few_shot=0):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.has_valid = has_valid
        self.few_shot =few_shot
        if llm:
            self.train_path = f'k_shot/n_way_{few_shot}_shot_train_2.json' if few_shot else f'{self.dataset_name}_train_2.json'
            self.cot_train_path = f'k_shot/n_way_{few_shot}_shot_train_COT_2.json' if few_shot else f'{self.dataset_name}_train_COT_2.json'
        else:
            self.train_path = f'k_shot/n_way_{few_shot}_shot_train_2.json' if few_shot else f'{self.dataset_name}_train.json'

    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.train_path}',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json'
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_dev.json'})
        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)

        return datasets

    def load_llm_CoT(self):
        rationales = list()
        with open(f'{self.data_root}/{self.dataset_name}/{self.cot_train_path}','r', encoding='utf-8') as fr:
            for line in fr:
                json_line = json.loads(line)
                rationale = json_line['CoT']
                if self.dataset_name in ['NCBI', 'BC5CDR']:
                    entities = json_line['entities']
                    label = [entity['span'] for entity in entities]
                    label = "disease* " + " disease* *disease ".join(label) + " *disease"
                elif self.dataset_name in ['MedNLI']:
                    label = json_line['label']
                elif self.dataset_name in ['MedQA']:
                    label = json_line['answer']
                elif self.dataset_name in ['i2b2_2010', 'n2c2_2018_track2']:
                    label = json_line['relation']['relation_type']
                else:
                    label = json_line['relation']
                explain = label + " explanation: " + rationale
                rationales.append(explain)

        return rationales

    def _post_process(self, datasets):
        raise NotImplementedError


class NERDatasetLoader(DatasetLoader):
    def __init__(self,dataset_name, llm, few_shot=None):
        self.has_valid = False
        self.task_prefix = "disease: "
        self.explain_prefix = "explain disease: "
        super().__init__(dataset_name, self.has_valid, llm, few_shot)

    def _process_label(self, sentence, entities):
        label = sentence.split()
        for entity in entities:
            start = entity['start']
            end = entity['end']
            e_type = entity['e_type'].lower()
            label[start] = '{}* '.format(e_type) + label[start]
            label[end - 1] = label[end - 1] + ' *{}'.format(e_type)
        label = ' '.join(label)
        return label

    def _post_process(self, datasets):

        def prepare_input(example):
            sentence = example['sentence']
            entities = example['entities']
            label = self._process_label(sentence, entities)
            example['input'] = sentence
            example['task_prefix'] = self.task_prefix
            example['explain_prefix'] = self.explain_prefix
            example['label'] = label

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['sentence', 'entities'])

        return datasets



class REDatasetLoader(DatasetLoader):
    def __init__(self,dataset_name, llm, few_shot=0):
        self.has_valid = False
        self.dataset_name = dataset_name
        if dataset_name  == 'AIMed':
            self.task_prefix = "ppi: "
            self.explain_prefix = "explain ppi: "
        elif dataset_name == 'HPRD50':
            self.task_prefix = "ppi: "
            self.explain_prefix = "explain ppi: "
        elif dataset_name == 'i2b2_2010':
            self.task_prefix = "re: "
            self.explain_prefix = "explain re: "
        elif dataset_name == 'n2c2_2018_track2':
            self.task_prefix = "re: "
            self.explain_prefix = "explain re: "
        else:
            raise ValueError
        super().__init__(dataset_name, self.has_valid, llm, few_shot)

    def _post_process(self, datasets):

        def prepare_input(example):
            sentence = example['sentence']
            if self.dataset_name in ['n2c2_2018_track2', 'i2b2_2010']:
                relation = example['relation']
                head = relation['head']
                tail = relation['tail']
                sentence = f'{sentence} head entity: {head} tail entity: {tail}'
                label = relation['relation_type']
            else:
                label = example['relation']

            example['input'] = sentence
            example['task_prefix'] = self.task_prefix
            example['explain_prefix'] = self.explain_prefix
            example['label'] = label

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['sentence', 'relation'])

        return datasets


class NLIDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, llm, few_shot=None):
        self.has_valid = True
        self.task_prefix = "mednli "
        self.explain_prefix = "explain mednli "
        super().__init__(dataset_name, self.has_valid, llm, few_shot)


    def _post_process(self, datasets):
        def prepare_input(example):
            Premise = example['Premise']
            Hypothesis = example['Hypothesis']
            input = f'premise: {Premise}. hypothesis: {Hypothesis}'
            example['input'] = input
            example['task_prefix'] = self.task_prefix
            example['explain_prefix'] = self.explain_prefix

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['Premise', 'Hypothesis'])

        return datasets


class QADatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, llm, few_shot=None):
        self.has_valid = True
        self.task_prefix = "qa: "
        self.explain_prefix = "explain qa "
        super().__init__(dataset_name, self.has_valid, llm, few_shot)

    def _post_process(self, datasets):
        def prepare_input(example):
            question = example['question']
            input = f"question: {question}.\nAnswer choices:\n(a): {example['choices1']}\n(b): {example['choices2']}\n(c): {example['choices3']}\n(d): {example['choices4']}"

            example['input'] = input
            example['task_prefix'] = self.task_prefix
            example['explain_prefix'] = self.explain_prefix
            example['label'] = example['answer']
            example['idx'] = example['question_id']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['question_id','question', 'answer', 'choices1', 'choices2', 'choices3', 'choices4'])

        return datasets


def load_llm_CoT(loader, datasets):
    train_cot = loader.load_llm_CoT()
    datasets['train'] = datasets['train'].add_column('rationale', train_cot)
    return datasets


def load_datasets(dataset_name, few_shot, llm):

    if dataset_name in ['NCBI', 'BC5CDR']:
        dataset_loader = NERDatasetLoader(dataset_name=dataset_name, llm=llm, few_shot=few_shot)
        datasets = dataset_loader.load_from_json()
        if llm:
            datasets = load_llm_CoT(dataset_loader, datasets)

        return datasets
    elif dataset_name in ['AIMed', 'HPRD50', 'i2b2_2010', 'n2c2_2018_track2']:
        dataset_loader = REDatasetLoader(dataset_name=dataset_name, llm=llm, few_shot=few_shot)
        datasets = dataset_loader.load_from_json()
        if llm:
            datasets = load_llm_CoT(dataset_loader, datasets)

        return datasets
    elif dataset_name in ['MedNLI']:
        dataset_loader = NLIDatasetLoader(dataset_name=dataset_name, llm=llm, few_shot=few_shot)
        datasets = dataset_loader.load_from_json()
        if llm:
            datasets = load_llm_CoT(dataset_loader, datasets)

        return datasets
    elif dataset_name in ['MedQA']:
        dataset_loader = QADatasetLoader(dataset_name=dataset_name, llm=llm, few_shot=few_shot)
        datasets = dataset_loader.load_from_json()
        if llm:
            datasets = load_llm_CoT(dataset_loader, datasets)

        return datasets

    else:
        raise ValueError




