import json
import numpy as np


def precision_score(_y_truth, _y_preds):
    tp = [y for y in _y_preds if y in _y_truth]
    precision = float(len(tp)) / len(_y_preds) if len(_y_preds) > 0 else 1
    return precision


drug_path_list = [("alprazolam", "test_alprazolam.json"),
                  ("ibuprofen", "test_ibuprofen.json"),
                  ("levothyroxine", "test_levothyroxine.json"),
                  ("metoformin", "test_metoformin.json"),
                  ("omeprazole", "test_omeprazole.json")]

for drug, result_path in drug_path_list:
    neural_precisions, umls_precisions, neat_precisions = [], [], []
    with open(result_path, "rb") as f:
        result = json.load(f)
    for doc_id, doc_data in result.items():
        y_truth, y_pred_umls, y_pred_neural, y_pred_neat \
            = doc_data["truth"], doc_data["umls"], doc_data["neural"], doc_data["neat"]
        neural_precisions.append(precision_score(y_truth, y_pred_neural))
        umls_precisions.append(precision_score(y_truth, y_pred_umls))
        neat_precisions.append(precision_score(y_truth, y_pred_neat))
    print("Drug: {}, UMLS: {}, Neural Extractor: {}, NEAT: {}"
          .format(drug, np.mean(neural_precisions), np.mean(umls_precisions), np.mean(neat_precisions)))


