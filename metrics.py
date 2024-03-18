import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support



def has_duplicate(tmp_list):
    if tmp_list == []:
        return False

    if type(tmp_list[0]) == str:
        if len(tmp_list) == len(set(tmp_list)):
            return False
        else:
            return True

    if type(tmp_list[0]) == list:
        tmp = []
        for t in tmp_list:
            if t not in tmp:
                tmp.append(t)
        if len(tmp_list) == len(tmp):
            return False
        else:
            return True

def get_correct_list_from_response_list(target_list, response_list):
    """
    target_list 和 response_list 均有可能包含重复的 item
    """

    res = []
    if not has_duplicate(response_list):
        res = [item for item in response_list if item in target_list]
    else:
        if not has_duplicate(target_list):
            # 去重
            uni_response_list = []
            for item in response_list:
                if item not in uni_response_list:
                    uni_response_list.append(item)
            res = [item for item in uni_response_list if item in target_list]
        else:
            res = []
            processed_item_list = []
            for item in response_list:
                if item not in processed_item_list:
                    processed_item_list.append(item)

                    num_item = response_list.count(item)
                    if num_item == 1:  # not duplicate
                        if item in target_list:
                            res.append(item)
                    else:  # duplicate
                        if item in target_list:
                            num_item_in_target = target_list.count(item)
                            num_item_correct = min([num_item, num_item_in_target])
                            res += [item] * num_item_correct

    return res



def compute_ner_metrics(sequence_true, sequence_pred):
    tp_ner_strict, fp_ner_strict, fn_ner_strict = 0, 0, 0
    assert len(sequence_true) == len(sequence_pred)
    for seq_true, seq_pred in zip(sequence_true, sequence_pred):
        entity_true_list = re.findall(r'disease\* (.*?) \*disease', seq_true)
        entity_pred_list = re.findall(r'disease\* (.*?) \*disease', seq_pred)
        strict_correct_list = get_correct_list_from_response_list(entity_true_list, entity_pred_list)

        tp_ner_strict += len(strict_correct_list)
        fp_ner_strict += len(entity_pred_list) - len(strict_correct_list)
        fn_ner_strict += len(entity_true_list) - len(strict_correct_list)

    precision = tp_ner_strict / (tp_ner_strict + fp_ner_strict) if (tp_ner_strict + fp_ner_strict) > 0 else 0
    recall = tp_ner_strict / (tp_ner_strict + fn_ner_strict) if (tp_ner_strict + fn_ner_strict) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_prf_text(tokenizer, NER=False):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred## predictions[0] predictions[1] 是一样的
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if NER:
            precision, recall, f1 = compute_ner_metrics(decoded_labels, decoded_preds)
            return {'precision': precision, 'recall': recall, 'f1': f1}
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=decoded_labels, y_pred=decoded_preds, average='micro'
            )
            return {'precision': precision, 'recall': recall, 'f1': f1}

    return compute_metrics


def compute_prf_text_aux(tokenizer, NER=False):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if NER:
            precision, recall, f1 = compute_ner_metrics(decoded_labels, decoded_preds)
            return {'precision': precision, 'recall': recall, 'f1': f1}
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=decoded_labels, y_pred=decoded_preds, average='micro'
                )
            return {'precision': precision, 'recall': recall, 'f1': f1}

    return compute_metrics




def compute_accuracy_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

def compute_accuracy_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics