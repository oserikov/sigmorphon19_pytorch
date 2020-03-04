import csv
import os



def extract_lang_pairs_from_data_paths(data_folder_path):
    lang_pairs = list(map(lambda x: tuple(x.split('--')),
                          os.listdir(data_folder_path)))
    return lang_pairs


def load_original_data(language_pair, task_folder):
    lang_pair_data = {}

    lang_pairs = extract_lang_pairs_from_data_paths(task_folder)

    for lang_pair in lang_pairs:
        if lang_pair != language_pair:
            continue

        lang_pair_folder_name = '--'.join(lang_pair)
        hi_lan = lang_pair[0]
        lo_lan = lang_pair[1]
        hi_resource_data_fn = f"{hi_lan}-train-high"
        lo_resource_data_fn = f"{lo_lan}-train-low"

        hi_resource_data_path = os.path.join(task_folder,
                                             lang_pair_folder_name,
                                             hi_resource_data_fn)

        lo_resource_data_path = os.path.join(task_folder,
                                             lang_pair_folder_name,
                                             lo_resource_data_fn)

        hi_resource_data = []
        with open(hi_resource_data_path, encoding="utf-8") as hi_res_tsv_f:
            hi_res_reader = csv.DictReader(hi_res_tsv_f,
                                           fieldnames=["lem", "wf", "tags"],
                                           delimiter='\t')
            for row in hi_res_reader:
                row["tags"] = row["tags"].split(';')
                hi_resource_data.append(row)

        lo_resource_data = []
        with open(lo_resource_data_path, encoding="utf-8") as lo_res_tsv_f:
            lo_res_reader = csv.DictReader(lo_res_tsv_f,
                                           fieldnames=["lem", "wf", "tags"],
                                           delimiter='\t')
            for row in lo_res_reader:
                row["tags"] = row["tags"].split(';')
                lo_resource_data.append(row)

        lang_pair_data[lang_pair] = {
            "high_lan": hi_lan,
            "low_lan": lo_lan,
            "high_data": hi_resource_data,
            "low_data": lo_resource_data
        }

    return lang_pair_data


def flatten_data(data_as_dict):
    flattened_data = []

    for lang_pair, lang_pair_data in data_as_dict.items():
        hi_lang = lang_pair_data["high_lan"]
        lo_lang = lang_pair_data["low_lan"]
        hi_lang_data = lang_pair_data["high_data"]
        lo_lang_data = lang_pair_data["low_data"]

        for data_entry in hi_lang_data:
            flattened_entry = data_entry
            flattened_entry["lang"] = hi_lang
            flattened_data.append(flattened_entry)

        for data_entry in lo_lang_data:
            flattened_entry = data_entry
            flattened_entry["lang"] = lo_lang
            flattened_data.append(flattened_entry)

    return flattened_data