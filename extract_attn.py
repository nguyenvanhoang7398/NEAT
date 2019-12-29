from constants import *
from utils import *
from train import evaluate
from tqdm import tqdm


cnn_result = read_json("output/benchmark/attention/neat_cnn_full_15_extracted.json")
umls_result = read_json("output/benchmark/umls_docs_se.json")
adr_result = read_json("output/benchmark/adr_docs_se.json")
test_id = read_json("data/pod/meta_100/train_test.json")["test"]
doc_ses_map = read_json(OUTPUT_DOC_SE_MAP_PATH)
doc_drug_map = read_json(OUTPUT_DOC_DRUG_MAP_PATH)
drug_se_map = read_json(OUTPUT_DRUG_SE_MAP_PATH)
doc_dir = OUTPUT_DOC_DIR
output_example_path = "examples.json"

examples = {}

for doc_id in tqdm(test_id, desc="Extracting examples"):
    if doc_id not in cnn_result or len(cnn_result[doc_id]) == 0:
        continue
    if doc_id not in umls_result or len(umls_result[doc_id]) == 0:
        continue
    if doc_id not in adr_result or len(adr_result[doc_id]) == 0:
        continue
    true_ses = doc_ses_map[doc_id]
    cnn_output = evaluate([true_ses], [cnn_result[doc_id]], num_labels=0, vector=False)
    cnn_precision = cnn_output["precision"]
    umls_output = evaluate([true_ses], [umls_result[doc_id]], num_labels=0, vector=False)
    umls_precision = umls_output["precision"]
    adr_output = evaluate([true_ses], [adr_result[doc_id]], num_labels=0, vector=False)
    adr_precision = adr_output["precision"]
    if cnn_precision > adr_precision > umls_precision:
        doc_path = os.path.join(doc_dir, "{}.json".format(doc_id))
        doc_content = read_json(doc_path)
        drugs = doc_drug_map[doc_id]
        drug_se = {drug: drug_se_map[drug] for drug in drugs}
        examples[doc_id] = {
            "truth": drug_se,
            "cnn": (cnn_result[doc_id], cnn_precision),
            "umls": (umls_result[doc_id], umls_precision),
            "adr": (adr_result[doc_id], adr_precision),
            "doc_content": doc_content
        }
write_json(examples, output_example_path)
