
import ast
import time
import openai
import backoff
import json
import logging
from sklearn.metrics import precision_recall_fscore_support


openai.api_type = "azure"
openai.api_base = "https://zfxs.openai.azure.com/"
openai.api_version = ""
openai.api_key = ""


Log_File = "RE.log"

# logging.basicConfig(filename=Log_File)
formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
sh = logging.StreamHandler()
fh = logging.FileHandler(filename=Log_File)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(fh)
logger.addHandler(sh)


### for exponential backoff
@backoff.on_exception(backoff.expo, \
                      (openai.error.RateLimitError,
                       openai.error.APIConnectionError,
                       openai.error.APIError,
                       openai.error.ServiceUnavailableError))
def cot_generation(content):
    response = openai.ChatCompletion.create(
                    engine="chenpeng",
                    messages=[{"role": "system", "content": content}],
                    temperature=0.7,
                    max_tokens=256,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
    cot = response['choices'][0]['message']['content']

    return cot



def basic_runner(content, max_retry=3, time_interval=30):
    retry = 0
    get_result = False
    cot = ''
    error_msg = ''

    while not get_result:
        try:
            cot = cot_generation(content)
            get_result = True
        except openai.error.RateLimitError as e:
            if e.user_message == 'You exceeded your current quota, please check your plan and billing details.':
                raise e
            elif retry < max_retry:
                time.sleep(time_interval)
                retry += 1
            else:
                error_msg = e.user_message
                break
        except Exception as e:
            raise e
    return get_result, cot, error_msg


def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False)
    f.write('\n')
    f.close()




class GPT_for_extract():
    def __init__(self, test_RE_path, save_RE_path, error_file_path, relation_des):
        self.test_RE_path = test_RE_path
        self.save_RE_path = save_RE_path
        self.error_file_path = error_file_path

        self.rel_pairs = '@DRUG1$ and @DRUG2$' if 'DDI' in self.test_RE_path else '@PROTEIN1$ and @PROTEIN2$'
        with open(relation_des, 'r') as file:
            relation_des = file.read()
        self.rel_dict = list(json.loads(relation_des)['rel_type'].keys())

    def predict_RE(self):
        counter = 0
        with open(self.test_RE_path, 'r', encoding='utf-8') as fr, open(self.save_RE_path, 'a', encoding='utf-8') as fw:
            for line in fr:
                json_line = json.loads(line)
                sentence = json_line['sentence']
                relation_type = json_line['relation']

                instruction = "Consider predefined relation types {}, classify relation between {} pairs from the given sentence and " \
                              "give the final answer with the format of ['relation_type'].".format(set(self.rel_dict), self.rel_pairs)
                prompt = "\nCoT: Let's think step by step."
                content = instruction + '\n' + 'sentence: ' + sentence + prompt
                print(content)
                print("===" * 20)
                exit(0)
                try:
                    get_result, CoT, error_msg = basic_runner(content)
                    if not get_result: counter += 1
                    new_json_line = {"idx": json_line['idx'], "sentence": sentence, "relation": relation_type, "response": CoT}
                    json.dump(new_json_line, fw)
                    fw.write('\n')
                except Exception as e:
                    write_json(json_line, self.error_file_path)
                    logger.warning(
                        f"an error raised when predicting (sentence: {json_line['idx']}). "
                        f"ERROR: {getattr(e.__class__, '__name__')}:{str(e)}"
                    )

    def extract_predict(self, response):
        def get_list_by_string(list_str):
            try:
                res_list = ast.literal_eval(list_str)
            except:
                res_list = []
            finally:
                return res_list

        response = response.split("\n")[-1]
        start_idx = response.find('[')
        end_idx = response.find(']')
        answer = response[start_idx: end_idx + 1]
        answer = get_list_by_string(answer)
        return answer

    def evaluate_RE(self):
        gold_labels = []
        predict_labels = []
        with open(self.save_RE_path, "r", encoding="utf-8") as fr:
            for line in fr:
                json_line = json.loads(line)
                label = json_line['relation']
                response = json_line['response']
                predict_label = self.extract_predict(response)
                gold_labels.append(label)
                predict_labels.append(predict_label)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=gold_labels, y_pred=predict_labels, average='micro'
        )
        dataset_name = self.test_RE_path.split('/')[-1][:-5]

        print("dataset_name:", dataset_name)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        return {'precision': precision, 'recall': recall, 'f1': f1}








if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_test_path', type=str, default='AIMed', choices=['AIMed', 'HPRD50'])
    parser.add_argument('--save_data_test_path', type=str, default='AIMed_test_from_gpt3,json')
    parser.add_argument('--error_file_path', type=str, default='AIMed_test_from_gpt3_error,json')
    parser.add_argument('--rel_des', type=str, default='AIMed_des.json')

    args = parser.parse_args()

    chatglm_extract_re = GPT_for_extract(args.data_test_path, args.save_data_test_path, args.error_file_path, args.rel_des)
    chatglm_extract_re.predict_RE()
    chatglm_extract_re.evaluate_RE()





