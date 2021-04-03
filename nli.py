import os
import random

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
import pandas as pd
import plac
from nltk.corpus import verbnet as vn
from sklearn.model_selection import train_test_split
import pdb

import properties
import numpy as np

ADJ = [
    "aggressive",
    "agreeable",
    "ambitious",
    "brave",
    "calm",
    "delightful",
    "eager",
    "faithful",
    "gentle",
    "happy",
    "jolly",
    "kind",
    "lively",
    "nice",
    "obedient",
    "polite",
    "proud",
    "silly",
    "thankful",
    "victorious",
    "witty",
    "wonderful",
    "zealous",
    "angry",
    "bewildered",
    "clumsy",
    "defeated",
    "embarrassed",
    "fierce",
    "grumpy",
    "helpless",
    "itchy",
    "jealous",
    "lazy",
    "mysterious",
    "nervous",
    "obnoxious",
    "panicky",
    "pitiful",
    "repulsive",
    "scary",
    "thoughtless",
    "uptight",
    "worried",
    "attractive",
    "clean",
    "dazzling",
    "drab",
    "elegant",
    "fancy",
    "fit",
    "glamorous",
    "handsome",
    "muscular",
    "plain",
    "scruffy",
    "shapely",
    "short",
    "skinny",
    "stocky"
]

def get_template(config):
    editor = Editor()
    entailment = config["entailment"]
    overlap = config["overlap"]
    if entailment and overlap:
        return [
            editor.template({'premise': '{first_name} is {adj1} but not {adj2}.',
                       'hypothesis': '{first_name} is {adj1}.'},
                      labels=1,
                      adj=ADJ,
                      remove_duplicates=True),
            editor.template({'premise': '{first_name} is {adj1} but not {adj2}.',
                       'hypothesis': '{first_name} is not {adj2}.'},
                      labels=1,
                      adj=ADJ,
                      remove_duplicates=True)]
    elif overlap:
        return [
            editor.template({'premise': '{first_name} is {adj1} but not {adj2}.',
                       'hypothesis': '{first_name} is {adj2}.'},
                      labels=0,
                      adj=ADJ,
                      remove_duplicates=True)
        ]
    else:
        return [
            editor.template({'premise': '{first_name} is {adj1} but not {adj2}.',
                       'hypothesis': '{first_name} is not {adj1}.'},
                      labels=0,
                      adj=ADJ,
                      remove_duplicates=True)
        ]

def generate(tpl):
    toks = []
    for t in tpl.split():
        if t in grammar:
            toks.append(random.choice(grammar[t]))
        else:
            toks.append(t)
    new = " ".join(toks)
    if not new == tpl:
        # print(new)
        return generate(new)
    return new + " ."


def make_dataset(section_to_count, template, easy_feature):
    dataset = []

    config_path = os.path.join("data/nli", f"{template}_{easy_feature}.csv")
    section_to_configs = properties.get_config(config_path)

    for section in section_to_count:
        templates = []
        section_data = []
        for config in section_to_configs[section]:
            for template in get_template(config):
                section_data.extend(zip(template.data[:section_to_count[section]], template.labels[:section_to_count[section]]))
        
        random.shuffle(section_data)

        for pair, label in section_data:
            dataset.append(dict(**pair, label=label, section=section))
    return dataset


def make_tsv_line(el):
    return "{}\t{}\t{}\t{}\n".format(el["premise"], el["hypothesis"], el["section"], el["label"])


@plac.opt("template", "template to use", choices=["base"])
@plac.opt(
    "weak", "weak feature to use", choices=["overlap"]
)
def main(template="base", weak="overlap"):
    random.seed(42)
    section_size = 1000
    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/nli_{template}_{weak}/"):
        os.mkdir(f"./properties/nli_{template}_{weak}/")

    dataset = make_dataset(
        # 1250 to handle duplicates.
        {
            "both": section_size + 1250,
            "neither": section_size + 1250,
            "weak": section_size + 1250,
            "strong": 0 * (section_size + 250),
        },
        template,
        weak,
    )
    all_df = pd.DataFrame(dataset).drop_duplicates()

    base_df = all_df[all_df.section.isin({"both", "neither"})]
    train_base, test_base = train_test_split(base_df, test_size=0.5)

    counterexample_df = all_df[all_df.section.isin({"weak"})]
    train_counterexample, test_counterexample = train_test_split(
        counterexample_df, test_size=0.5
    )
    rates = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    properties.generate_property_data(
        "nli_{}_{}".format(template, weak),
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        section_size,
        rates,
    )


if __name__ == "__main__":
    plac.call(main)
