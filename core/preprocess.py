import pandas as pd 

from sklearn.preprocessing import MultiLabelBinarizer

import re
import os
from pathlib import Path
from core.utils import Config

def get_input_data()->pd.DataFrame:

    def load(file_path: str):
        df = pd.read_csv(file_path, skipinitialspace=True)
        df.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
        return df

    df1 = load(os.path.join('data','raw','AppGallery.csv'))
    df2 = load(os.path.join('data','raw','Purchasing.csv'))
    df = pd.concat([df1, df2])

    # Ensure that the columns are strings
    df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].values.astype('U')
    df[Config['TICKET_SUMMARY']] = df[Config['TICKET_SUMMARY']].values.astype('U')
    
    # Label column
    df['labels'] = df[Config['TYPE_COLS']].apply(
        lambda row:[str(label) for label in row if pd.notnull(label) and label != ''],
        axis = 1
    )

    return df

def _get_customer_support_template():
    """
    Returns a dictionary with customer support templates for different languages.
    """
    return {
        "english":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            [r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            [r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            [r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

def _compile_patterns(template: list[str]) -> str:
    """
    Compiles a list of regex patterns into a single pattern.
    """
    return '|'.join(f"({p})" for p in sum(template.values(), []))

def _compile_split_patterns() -> str:
    """
    Compiles a list of regex patterns for splitting interaction content.
    """
    pattern = [
        r"From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:",
        r"On.{30,60}wrote:",
        r"Re\s?:|RE\s?:",
        r"\*\*\*\*\*\(PERSON\) Support issue submit",
        r"\s?\*\*\*\*\*\(PHONE\)*$"
    ]
    return '|'.join(f'{p}' for p in pattern)

def de_duplication(data: pd.DataFrame):
    data["ic_deduplicated"] = ""
    
    cu_template = _get_customer_support_template()
    cu_pattern = _compile_patterns(cu_template)
    split_pattern = _compile_split_patterns()

    # -------- start processing ticket data
    for ticket_id in data["Ticket id"].unique():
        df_ticket = data[data["Ticket id"] == ticket_id]
        ic_set, deduped_contents = set(), []

        for ic in df_ticket[Config['INTERACTION_CONTENT']]:
            segments = [re.sub(cu_pattern, "", re.sub(split_pattern, "", part.strip()))
                        for part in re.split(split_pattern, ic) if part]
            new_parts = [seg + "\n" for seg in segments if seg and seg not in ic_set and not ic_set.add(seg)]
            deduped_contents.append(' '.join(new_parts))

        data.loc[data["Ticket id"] == ticket_id, "ic_deduplicated"] = deduped_contents
    # -------- end processing ticket data

    data[Config['INTERACTION_CONTENT']] = data['ic_deduplicated']
    return data.drop(columns=['ic_deduplicated'])

def noise_remover(df: pd.DataFrame):
    def clean_column(series: pd.Series, noise_patterns) -> pd.Series:
        for pattern in noise_patterns:
            series = series.replace(pattern, " ", regex=True)
        return series.replace(r'\s+', ' ', regex=True).str.strip()

    summary_col, content_col = Config['TICKET_SUMMARY'], Config['INTERACTION_CONTENT']
    noise_header = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|\[|\]|null|nan|(support.pt 自动回复:)"
    df[summary_col] = clean_column(df[summary_col].str.lower(), [noise_header])
    df[content_col] = df[content_col].str.lower()

    noise_body = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)" ,
        r"(january|february|...|december)",
        r"(jan|feb|...|dec)",
        r"(monday|...|sunday)",
        r"\d{2}(:|.)\d{2}",
        r"xxxxx@xxxx\.com|\*{5}\([a-z]+\)",
        r"dear( customer| user)?", r"hello|hi( there)?|good morning",
        r"thank you.*", r"we apologize.*", r"sent from my huawei.*",
        r"original message", r"customer support team",
        r".*se is a company incorporated.*", r".*mobile services.*",
        r"canada, australia,.*", r"\d+", r"[^0-9a-zA-Z]+",
        r"(\s|^)\.(\s|$)"
    ]
    df[content_col] = clean_column(df[content_col], noise_body)

    valid_labels = df.y1.value_counts()[lambda x: x > 10].index
    return df[df.y1.isin(valid_labels)]

def translate_to_en(texts:list[str]):
    from langdetect import detect_langs
    from transformers import pipeline
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    model_id = "facebook/m2m100_418M"  # M2M-100 model for multilingual translation
    tokenizer = M2M100Tokenizer.from_pretrained(model_id)
    model = M2M100ForConditionalGeneration.from_pretrained(model_id)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, tgt_lang="en"
                          , src_lang="auto")
    result = []

    for text in texts:
        if text == "":
            result.append(text)
            continue

        # Detect language
        lang = detect_langs(text)[0].lang
        if lang == "en":
            result.append(text)
        else:
            # Translate to English
            translation = translator(text, src_lang=lang, tgt_lang="en")
            result.append(translation[0]['translation_text'])

    return result

def multi_label_en(df: pd.DataFrame):
    
    mlb = MultiLabelBinarizer()
    y= mlb.fit_transform(df['labels'])
    print(list(mlb.classes_))

    return y